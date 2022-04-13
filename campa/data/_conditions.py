# functions for converting conditions to strings or one-hot encoded vectors
from typing import Any, List, Tuple, Optional, MutableMapping
import logging

import numpy as np

from campa.constants import get_data_config


def get_one_hot(targets: np.ndarray, nb_classes: int) -> np.ndarray:
    res: np.ndarray = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def get_combined_one_hot(arrs: List[np.ndarray]) -> np.ndarray:
    if len(arrs) != 2:
        raise NotImplementedError(f"combine {len(arrs)} arrs")
    mask = (~np.isnan(arrs[0][:, 0])) & (~np.isnan(arrs[1][:, 0]))
    n1 = arrs[0].shape[1]
    n2 = arrs[1].shape[1]
    targets = np.zeros(len(arrs[0]), dtype=np.uint8)
    targets[mask] = np.argmax(arrs[0][mask], axis=1) + n1 * np.argmax(arrs[1][mask], axis=1)
    res = get_one_hot(targets, n1 * n2)
    res[~mask] = np.nan
    return res


def convert_condition(
    arr: np.ndarray, desc: str, one_hot: bool = False, data_config: Optional[Any] = None
) -> np.ndarray:
    """Convert condition array according to desc."""
    log = logging.getLogger("convert_condition")
    if data_config is None:
        log.warning("using default data config")
        data_config = get_data_config()
    # check if is a condition that can convert:
    if desc in data_config.CONDITIONS.keys():
        cur_conditions = data_config.CONDITIONS[desc]
        # need to go from str to numbers or the other way?
        if np.isin(arr, cur_conditions).any():
            log.info(f"Converting condition {desc} to numbers")
            conv_arr = np.zeros(arr.shape, dtype=np.uint8)
            if "UNKNOWN" in cur_conditions:
                cur_conditions.append(cur_conditions.pop(cur_conditions.index("UNKNOWN")))
                conv_arr[~np.in1d(arr, cur_conditions)] = cur_conditions.index("UNKNOWN")
            for i, c in enumerate(cur_conditions):
                conv_arr[arr == c] = i
            if one_hot:
                conv_arr = get_one_hot(conv_arr, len(cur_conditions))
        else:
            log.info(f"Converting condition {desc} to strings")
            conv_arr = np.zeros(arr.shape, dtype=np.object)  # type: ignore[attr-defined]
            for i, c in enumerate(cur_conditions):
                conv_arr[arr == i] = c
        return conv_arr
    else:
        log.info(f"Not converting condition {desc} (is regression)")
        return arr


def process_condition_desc(desc: str) -> Tuple[str, Optional[str]]:
    postprocess = None
    for proc in ["_one_hot", "_bin_3", "_lowhigh_bin_2", "_zscore"]:
        if proc in desc:
            desc = desc.replace(proc, "")
            postprocess = proc[1:]
    return desc, postprocess


def get_bin_3_condition(
    cond: np.ndarray, desc: str, cond_params: MutableMapping[str, Any]
) -> Tuple[np.ndarray, List[float]]:
    """
    look for desc_bin_3_quantile kwarg specifying the quantile.
    If not present, calculate the quantiles based on cond.
    Then bin cond according to quantiles
    """
    # bin in .33 and .66 quantiles (3 classes)
    if cond_params.get(f"{desc}_bin_3_quantile", None) is not None:
        q = cond_params[f"{desc}_bin_3_quantile"]
    else:
        q = np.quantile(cond, q=(0.33, 0.66))
        cond_params[f"{desc}_bin_3_quantile"] = list(q)
    cond_bin = np.zeros_like(cond).astype(int)
    cond_bin[cond > q[0]] = 1
    cond_bin[cond > q[1]] = 2
    return cond_bin, list(q)


def get_lowhigh_bin_2_condition(
    cond: np.ndarray, desc: str, cond_params: MutableMapping[str, Any]
) -> Tuple[np.ndarray, List[float]]:
    # bin in 4 quantiles, take low and high TR cells (2 classes)
    # remainder of cells has nan values - can be filtered out later
    if cond_params.get(f"{desc}_lowhigh_bin_2_quantile", None) is not None:
        q = cond_params[f"{desc}_lowhigh_bin_2_quantile"]
    else:
        q = np.quantile(cond, q=(0.25, 0.75))
        cond_params[f"{desc}_lowhigh_bin_2_quantile"] = list(q)
    cond_bin = np.zeros_like(cond).astype(int)
    cond_bin[cond > q[1]] = 1
    return cond_bin, list(q)


def get_zscore_condition(
    cond: np.ndarray, desc: str, cond_params: MutableMapping[str, Any]
) -> Tuple[np.ndarray, List[float]]:
    # z-score TR
    # contiinous nbr, normalizes it
    # should work only after split up
    if cond_params.get(f"{desc}_mean_std", None) is not None:
        tr_mean, tr_std = cond_params[f"{desc}_mean_std"]
    else:
        tr_mean, tr_std = cond.mean(), cond.std()
        cond_params[f"{desc}_mean_std"] = [tr_mean, tr_std]
    cond = (cond - tr_mean) / tr_std
    return cond, [tr_mean, tr_std]
