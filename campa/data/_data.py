from copy import copy, deepcopy
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    Mapping,
    Iterable,
    Optional,
    Sequence,
    MutableMapping,
)
from logging import getLogger
import os
import json

import numpy as np
import pandas as pd
import anndata as ad

from campa.pl import annotate_img
from campa.constants import campa_config
from campa.data._img_utils import pad_border, BoundingBox
from campa.data._conditions import (
    get_one_hot,
    convert_condition,
    get_bin_3_condition,
    get_combined_one_hot,
    get_zscore_condition,
    process_condition_desc,
    get_lowhigh_bin_2_condition,
)


class ImageData:
    """
    Stub for future ImageData class.
    """

    def __init__(self, metadata, channels, imgs=None, seg_imgs=None, **kwargs):
        self.metadata = metadata
        self.channels = channels
        self.seed = kwargs.get("seed", 42)
        self.rng = np.random.default_rng(seed=self.seed)
        self.log = getLogger(self.__class__.__name__)
        self.imgs = imgs
        self.seg_imgs = seg_imgs

    @classmethod
    def from_dir(cls, data_dir):
        """Read metadata and images (lazy) and call constructor."""

    def get_obj_img(self, obj_id, kind="data"):
        """Crop and return object images for different kinds: data, mask (object mask)."""


class MPPData:
    """
    Pixel-level data representation.

    Backed by on-disk numpy and csv files containing intensity information per channel and metadata
    for each pixel.
    When possible, the on-disk numpy files are loaded lazily using :func:`np.memmap`.

    Parameters
    ----------
    metadata
        Cell-level metadata. Needs to contain at least an `data_config.OBJ_ID` column, which contains
        cell identifiers that are used in the `obj_ids` data to map pixels to cells.
    channels
        Channel-level metadata.
        The first column is assumed to be the index, second column the name of the channel.
    data
        Dictionary containing pixel-level data, at least containing the required keys: `x`, `y`, `obj_ids`
        (spatial coordinates for every pixel, and assignment of pixels to objects (cells)).
        If `mpp` (per-channel pixel intensity information) is not present,
        is replaced with zero-value array of shape: `#pixels x 1 x 1 x #channels`.
    data_config
        Name of the data_config file registered in `campa_config.data_configs`.
    seed
        Random seed for subsampling and subsetting.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        channels: pd.DataFrame,
        data: Dict[str, np.ndarray],
        data_config: str,
        seed: int = 42,
    ):
        # set up logger
        self.log = getLogger(self.__class__.__name__)
        # set up rng
        self.seed = seed
        # TODO change to default_rng as soon as have reproduced legacy datasets
        self.rng = np.random.RandomState(seed=self.seed)
        # save attributes
        self.data_config_name = data_config
        self.data_config = campa_config.get_data_config(self.data_config_name)
        self.channels: pd.DataFrame = channels
        """
        Intensity channels.
        """
        self._data = data
        # data dir and base dir
        self.data_dir: Union[str, None] = None
        self.base_dir: Union[str, None] = None

        for required_key in ["x", "y", "obj_ids"]:
            assert required_key in data.keys(), f"required key {required_key} missing from data"
        # subset metadata to obj_ids in data
        self.metadata: pd.DataFrame = metadata[metadata[self.data_config.OBJ_ID].isin(np.unique(self.obj_ids))]
        """
        Object (cell) level metadata (e.g. perturbation, cell-cycle, etc).
        """
        # add neighbor dimensions to mpp if not existing
        if len(self.mpp.shape) == 2:
            self._data["mpp"] = self.mpp[:, np.newaxis, np.newaxis, :]
        self.log.info(f"Created new: {self.__str__()}")

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str,
        data_config: str,
        mode: str = "r",
        base_dir: Optional[str] = None,
        keys: Iterable[str] = (),
        optional_keys: Iterable[str] = ("mpp", "labels", "latent", "conditions"),
        **kwargs: Any,
    ) -> "MPPData":
        """
        Initialise :class:`MPPData` from directory.

        Read data from `key.npy` for each `key` in `keys`.
        If present, will also read `key.npy` for each `key` in `optional_keys`.

        The information can be spread out over a chain of directories.
        In this case, a `mpp_params.json` file in `data_dir` indicates that more
        data can be found in `base_data_dir` (defined in `mpp_params`).
        First, the data from `base_data_dir` is loaded, and then the
        remaining information from `data_dir` is added.

        Each `data_dir` along this chain has to contain at least
        `x.npy`, `y.npy`, `obj_ids.npy`, `metadata.csv` and `channels.csv`.

        Parameters
        ----------
        data_dir
            Path to the specific directory containing one set of `npy` and `csv` files as described above.
            Note that this path should be relative to `base_dir`,
            which is set to `data_config.DATA_DIR` by default.
        data_config
            Name of the data_config file registered in `campa_config.data_configs`.
        mode
            `mmap_mode` for :func:`np.load`. Set to None to load data in memory.
        base_dir
            Look for data in `base_dir/data_dir`. Default in `data_config.DATA_DIR`.
        keys
            Read data from `key.npy` for each `key` in `keys`.
        optional_keys
            If present, read `key.npy` for each `key` in `optional_keys`.
        kwargs
            Passed to :class:`MPPData`.
        """
        # load data_config
        data_config_inst = campa_config.get_data_config(data_config)
        if base_dir is None:
            base_dir = data_config_inst.DATA_DIR

        # mpp_params present?
        if os.path.isfile(os.path.join(base_dir, data_dir, "mpp_params.json")):
            # first, load base_mpp_data
            res_keys, res_optional_keys = _get_keys(keys, optional_keys, None)
            mpp_params = json.load(open(os.path.join(base_dir, data_dir, "mpp_params.json")))
            # print(f"!!!!!loading base mpp from {data_config_inst.DATA_DIR}, {mpp_params['base_data_dir']}")

            self = cls.from_data_dir(
                mpp_params["base_data_dir"],
                data_config=data_config,
                keys=res_keys,
                optional_keys=res_optional_keys,
                base_dir=data_config_inst.DATA_DIR,
                mode=mode,
                **kwargs,
            )
            # print(f"!!!!!adding from {base_dir}, {data_dir}")
            # second, add mpp_data
            res_keys, res_optional_keys = _get_keys(keys, optional_keys, self)
            self.add_data_from_dir(
                data_dir,
                keys=res_keys,
                optional_keys=res_optional_keys,
                base_dir=base_dir,
                mode=mode,
                subset=mpp_params["subset"],
            )
            self.log.info(f"Loaded data from {data_dir}, with base data from {mpp_params['base_data_dir']}")
            self.data_dir = data_dir
            self.base_dir = base_dir
            return self
        else:
            # have reached true base_dir, load data
            # read all data from data_dir
            # NOTE: when normally loading
            self = cls._from_data_dir(
                data_dir, data_config, mode=mode, base_dir=base_dir, keys=keys, optional_keys=optional_keys, **kwargs
            )
            self.log.info(f"Loaded data from {data_dir}.")
        return self

    @classmethod
    def _from_data_dir(
        cls,
        data_dir: str,
        data_config: str,
        mode: str = "r",
        base_dir: Optional[str] = None,
        keys: Iterable[str] = (),
        optional_keys: Iterable[str] = ("mpp", "labels", "latent", "conditions"),
        **kwargs: Any,
    ) -> "MPPData":
        """
        Read MPPData from directory, ignoring mpp_params.json.
        """
        # ensure that x,y,obj_ids are in keys
        keys = list(set(["x", "y", "obj_ids"] + list(keys)))
        # get base_dir
        if base_dir is None:
            base_dir = campa_config.get_data_config(data_config).DATA_DIR

        metadata = pd.read_csv(os.path.join(base_dir, data_dir, "metadata.csv"), index_col=0).reset_index(drop=True)
        channels = pd.read_csv(
            os.path.join(base_dir, data_dir, "channels.csv"),
            names=["channel_id", "name"],
            index_col=0,
        ).reset_index(drop=True)
        channels.index.name = "channel_id"
        # read npy data
        data = {}
        for fname in keys:
            data[fname] = _try_mmap_load(os.path.join(base_dir, data_dir, f"{fname}.npy"), mmap_mode=mode)
        for fname in optional_keys:
            d = _try_mmap_load(
                os.path.join(base_dir, data_dir, f"{fname}.npy"),
                mmap_mode=mode,
                allow_not_existing=True,
            )
            if d is not None:
                data[fname] = d
        # init self
        self = cls(metadata=metadata, channels=channels, data=data, data_config=data_config, **kwargs)
        self.data_dir = data_dir
        self.base_dir = base_dir
        return self

    @classmethod
    def concat(cls, objs: List["MPPData"]) -> "MPPData":
        """
        Concatenate multiple MPPData objects.

        All MPPDatas that should be concatenated need to have the same keys,
        use the same data_config, and same channels.

        Parameters
        ----------
        objs
            List of objects to be concatenated.

        Returns
        -------
        Concatenated MPPData objects.
        """
        # channels, data_config, and _data.keys() need to be the same
        for mpp_data in objs:
            assert (mpp_data.channels.name == objs[0].channels.name).all()
            assert mpp_data._data.keys() == objs[0]._data.keys()
            assert mpp_data.data_config_name == objs[0].data_config_name

        channels = objs[0].channels
        # concatenate metadata (pandas)
        metadata = pd.concat([mpp_data.metadata for mpp_data in objs], axis=0, ignore_index=True)
        # concatenate numpy arrays
        data = {}
        for key in objs[0]._data.keys():
            data[key] = np.concatenate([mpp_data._data[key] for mpp_data in objs], axis=0)

        self = cls(
            metadata=metadata,
            channels=channels,
            data=data,
            seed=objs[0].seed,
            data_config=objs[0].data_config_name,
        )
        self.log.info(f"Created by concatenating {len(objs)} MPPDatas")
        return self

    # --- Properties ---
    @property
    def mpp(self) -> np.ndarray:
        """
        Multiplexed pixel profiles.
        """
        if "mpp" not in self._data.keys():
            self.log.info("Setting mpp to empty array")
            self._data["mpp"] = np.zeros((len(self.x), 1, 1, len(self.channels)))
        return self._data["mpp"]

    @property
    def obj_ids(self) -> np.ndarray:
        """
        Object ids mapping pixels to objects (cells).
        """
        return self._data["obj_ids"]

    @property
    def x(self) -> np.ndarray:
        """
        Spatial x coordinates for each pixel.
        """
        return self._data["x"]

    @property
    def y(self) -> np.ndarray:
        """
        Spatial y coordinates for each pixel.
        """
        return self._data["y"]

    @property
    def latent(self) -> Union[np.ndarray, None]:
        """
        Latent space for each pixel.
        """
        return self.data("latent")

    def data(self, key: str) -> Union[np.ndarray, None]:
        """
        Information contained in MPPData.

        Parameters
        ----------
        key
            Identifier for data that should be returned.

        Returns
        -------
            Array-like data, or None if data could not be found.
        """
        if key in self._data.keys():
            return self._data[key]
        elif key == "center_mpp":
            return self.center_mpp
        return None

    @property
    def conditions(self) -> Union[np.ndarray, None]:
        """
        Condition information for each pixel.
        """
        return self.data("conditions")

    @property
    def has_neighbor_data(self) -> bool:
        """
        Flag indicating if neighbour information is contained in this object.
        """
        return (self.mpp.shape[1] != 1) and (self.mpp.shape[2] != 1)

    @property
    def center_mpp(self) -> np.ndarray:
        """
        ``#pixels x #channels`` array of centre pixel.
        """
        c = int(self.mpp.shape[1] // 2)
        return self.mpp[:, c, c, :]  # type: ignore[no-any-return]

    @property
    def unique_obj_ids(self) -> np.ndarray:
        """
        Return unique objects (cells) contained in this object.
        """
        return np.unique(self.obj_ids)  # type: ignore[no-any-return]

    def __str__(self) -> str:
        s = f"MPPData for {self.data_config_name} ({self.mpp.shape[0]} mpps with shape {self.mpp.shape[1:]}"
        s += f" from {len(self.metadata)} objects)."
        s += f" Data keys: {list(self._data.keys())}."
        return s

    # --- Saving ---
    def write(self, save_dir: str, save_keys: Iterable[str] = None, mpp_params: Mapping[str, Any] = None) -> None:
        """
        Write MPPData to disk.

        Save channels, metadata as `csv` and one `npy` file per entry in :meth:`MPPData.data`.

        Parameters
        ----------
        save_dir
            Full path to directory in which to save ``MPPData``.
        save_keys
            Only save these MPPData.data entries. "x", "y", "obj_ids" are always saved.
            If None, all MPPData.data entries are saved.
        mpp_params
            Parameter dict to be saved in ``save_dir/mpp_params.json``. Use if ``save_keys`` is not None.
            Should contain ``base_data_dir`` (relative to ``data_config.DATA_DIR``), and subset information
            for correct MPPData initialisation.

        Returns
        -------
        Nothing, saves data to disk.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_keys is None:
            save_keys = self._data.keys()
        else:
            if mpp_params is None or mpp_params.get("base_data_dir", None) is None:
                self.log.warning("Saving partial keys of mpp data without a base_data_dir to enable correct loading")
            else:
                # save mpp_params
                mpp_params = {
                    "base_data_dir": mpp_params["base_data_dir"],
                    "subset": mpp_params.get("subset", False),
                }
                json.dump(
                    mpp_params,
                    open(os.path.join(save_dir, "mpp_params.json"), "w"),
                    indent=4,
                )
        # add required save_keys
        save_keys = list(set(save_keys).union(["x", "y", "obj_ids"]))
        self.log.info(f"Saving mpp data to {save_dir} (keys: {save_keys})")
        for key in save_keys:
            np.save(os.path.join(save_dir, f"{key}.npy"), np.array(self._data[key]))
        self.channels.to_csv(os.path.join(save_dir, "channels.csv"), header=None)
        self.metadata.to_csv(os.path.join(save_dir, "metadata.csv"))

    def copy(self) -> "MPPData":
        """
        Copy MPPData.
        """
        deepcopy(self._data)
        deepcopy(self.metadata)
        mpp_copy = MPPData(
            metadata=deepcopy(self.metadata),
            channels=deepcopy(self.channels),
            data=deepcopy(self._data),
            seed=deepcopy(self.seed),
            data_config=deepcopy(self.data_config_name),
        )
        return mpp_copy

    # TODO Nastassya: ensure that each function logs what its doing with log.info
    # --- Modify / Add data ---
    def add_data_from_dir(
        self,
        data_dir: str,
        keys: Iterable[str] = (),
        optional_keys: Iterable[str] = (),
        subset: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add data to MPPData from ``data_dir``.

        Expects ``x``, ``y``, and ``obj_ids`` to be present in ``data_dir``.
        Assumes that the data in ``data_dir`` uses the same ``data_config`` as the present object.

        Parameters
        ----------
        data_dir
            Full path to the dir containing the numpy files to be loaded.
        keys
            Filenames of data that should be loaded
        optional_keys
            Filenames of data that could optionally be loaded if present
        subset
            If ``True``, subset :class:`MPPData` to data present in ``data_dir``.
        kwargs
            Passed to :func:`MPPData.from_data_dir`, i.e., `seed`.

        Returns
        -------
        Nothing, modifies :class:`MPPData` in place. Adds ``keys`` to :func:`MPPData.data`.

        Raises
        ------
        ValueError
            If ``subset`` is False, and data in ``data_dir`` differs from data in ``self``.
        """
        mpp_to_add = MPPData._from_data_dir(
            data_dir,
            keys=keys,
            optional_keys=optional_keys,
            data_config=self.data_config_name,
            **kwargs,
        )
        # check that obj_ids from mpp_to_add are the same / a subset of the current obj_ids
        if subset:
            assert set(self.unique_obj_ids).issuperset(mpp_to_add.unique_obj_ids)
            # subset self to the obj_ids in mpp_to_add
            self.subset(obj_ids=mpp_to_add.unique_obj_ids)
        # now, data should be exactly the same
        assert (self.obj_ids == mpp_to_add.obj_ids).all()
        assert (self.x == mpp_to_add.x).all()
        assert (self.y == mpp_to_add.y).all()
        # finally, add keys
        added_mpp = False
        for key in keys:
            self._data[key] = mpp_to_add._data[key]
            added_mpp = added_mpp if key != "mpp" else True
        for key in optional_keys:
            if mpp_to_add.data(key) is not None:
                # check if mpp is not only zeros
                if key == "mpp" and (mpp_to_add.data("mpp") == np.zeros_like(mpp_to_add.mpp)).all():
                    continue
                self._data[key] = mpp_to_add._data[key]
                added_mpp = added_mpp if key != "mpp" else True
        # update channels if added mpp
        if added_mpp:
            self.channels = mpp_to_add.channels
        self.log.info(f"Updated data to keys {list(self._data.keys())}")

    def train_val_test_split(self, train_frac: float = 0.8, val_frac: float = 0.1) -> List["MPPData"]:
        """
        Split along obj_ids for train/val/test split.

        Parameters
        -----------
        train_frac
            Fraction of objects in train split.
        val_frac
            Fraction of objects in val split. Needs to contain at least one object.
        Returns
        -------
        train, val, test
        """
        # TODO (maybe) adapt and ensure that have even val/test fractions from each well
        ids = self.unique_obj_ids.copy()
        self.rng.shuffle(ids)
        num_train = int(len(ids) * train_frac)
        num_val = int(len(ids) * val_frac)
        train_ids = ids[:num_train]
        val_ids = ids[num_train : num_train + num_val]
        test_ids = ids[num_train + num_val :]
        self.log.info(f"Splitting data in {len(train_ids)} train, {len(val_ids)} val, and {len(test_ids)} test objects")
        splits = []
        for split_ids in (train_ids, val_ids, test_ids):
            ind = np.in1d(self.obj_ids, split_ids)
            splits.append(self.apply_mask(ind, copy=True))
        return splits

    def prepare(self, params: Mapping[str, Any]) -> None:
        """
        Prepare MPP data according to given parameters (from :func:`campa.data.create_dataset`).

        After calling this function, you might want to call :meth:`MPPData.subsample` and
        :meth:`MPPData.add_neighorhood`.

        Parameters
        ----------
        params
            Parameter dict describing how to prepare MPPData. See :func:`campa.data.create_dataset`.
            Ignores parameters `data_dirs`, `split`, `subsampling`, and `neighborhood`.

        Returns
        -------
        Nothing, modifies MPPData in place.
        """
        # subset channels - should be done first, as normalise writes params wrt channel ordering
        if params.get("channels", None) is not None:
            self.subset_channels(params["channels"])
        # normalise
        if params["normalise"]:
            self.normalise(**params["normalise_kwargs"])
        # add conditions
        if params.get("condition", None) is not None:
            self.add_conditions(params["condition"], **params["condition_kwargs"])
        # subset (after conditions, bc subsetting might need conditions)
        if params["subset"]:
            self.subset(**params["subset_kwargs"])

    # --- Detailed fns used in prepare() ---
    # TODO: Nastassya write tests for this
    def add_conditions(
        self, cond_desc: Iterable[Union[List[str], str]], cond_params: MutableMapping[str, Any] = None
    ) -> None:
        """
        Add conditions using :attr:`MPPData.metadata` columns.

        ``cond_desc`` describes the conditions that should be added.
        It is a list of condition descriptions.
        Each condition is calculated separately and concatenated to form the resulting
        :attr:`MPPData.conditions` vector.
        Condition descriptions have the following format: `"{condition}(_{postprocess})"`.

        Condition values are obtained as follows:

        * look up condition in :attr:`MPPData.metadata`.
          If `condition` is described in `data_config.CONDITIONS`, map it to numerical values.
          Note that if there is an entry `UNKNOWN` in `data_config.CONDITIONS`, all unmapped values will be mapped
          to this class.
          If `condition` is not described in `data_config.CONDITIONS`, values are assumed to be continuous
          and stored as they are in the condition vector.
        * post-process conditions. `postprocess` can be one of the following values:

           - `lowhigh_bin_2`: Only for continuous values. Removes middle values. Bin all values in 4 quantiles,
              encodes values in
              the lowest quantile as one class and values in the high quantile as the second class (one-hot encoded),
              and set all values in-between set to NaN.
           - `bin_3`: Only for continuous values. Bin values in .33 and .66 quantiles and one-hot encode each value.
           - `zscore`: Only for continuous values. Normalise values by mean and std.
           - `one_hot`: Only for categorical values. One-hot encode values.

        For categorical descriptions, it is possible to pass a list of condition descriptions.
        This will return a unique one-hot encoded vector combining multiple conditions.

        This operation is performed in place.

        Parameters
        ----------
        cond_desc
            Conditions to be added.
        cond_params
            Can optionally contain precomputed quantiles or mean/string values. If no values are provided,
            this will be filled with computed quantiles or mean/string values.
            This is useful for using the same values to process conditions on e.g. train and test sets.
        Returns
        -------
        Nothing, adds :attr:`MPPData.conditions`.

        """
        self.log.info(f"Adding conditions: {cond_desc}")
        conditions = []
        for desc in cond_desc:
            cond = self.get_condition(desc, cond_params)
            conditions.append(cond)
        self._data["conditions"] = np.concatenate(conditions, axis=-1)

    def subset(
        self,
        frac: Optional[float] = None,
        num: Optional[int] = None,
        obj_ids: Optional[Union[np.ndarray, List[int]]] = None,
        nona_condition: bool = False,
        copy: bool = False,
        **kwargs: Any,
    ) -> "MPPData":
        """
        Object-level subsetting of MPPData.

        Several filters for subsetting can be defined:

        - subset to random fraction / number of objects
        - filtering by object ids
        - filtering by values in :attr:`MPPData.metadata`
        - filtering by condition values

        Restrict objects to those with specified value(s) for key in the metadata table

        Parameters
        ----------
        frac
            Fraction of objects to randomly subsample.
            Applied after the other subsetting
        num
            Number of objects to randomly subsample. ``frac`` takes precedence.
            Applied after the other subsetting.
        obj_ids
            Object ids to subset to.
        nona_condition
            If set to True,  all values having NaN conditions will be filtered out.
            Note that the way conditions are created allows one to
            e.g. leave only entries which values in the specified column were in the low
            and high quantiles, and filter out everything else.
        copy
            Return new MPPData object or modify in place.
        kwargs
            Keys are column names in the metadata table.
            Values (str or list of str) are allowed entries for that key in the metadata table for selected objects.
            NO_NAN is special token selecting all values except NaN.

        Returns
        -------
            Subsetted MPPData
        """
        selected_obj_ids = np.array(self.metadata[self.data_config.OBJ_ID])
        self.log.info(f"Before subsetting: {len(selected_obj_ids)} objects")
        # select ids based on pre-defined ids
        if obj_ids is not None:
            selected_obj_ids = np.array(obj_ids)
            self.log.info(f"Subsetting to {len(obj_ids)} objects")
        # select ids based on kwargs
        for key, value in kwargs.items():
            cur_metadata = self.metadata.set_index(self.data_config.OBJ_ID).loc[selected_obj_ids]
            assert key in cur_metadata.columns, f"provided column {key} was not found in the metadata table!"
            if value == "NO_NAN":
                mask = ~cur_metadata[key].isnull()  # TODO check!
                self.log.info(f"Subsetting to NO_NAN {key}: {sum(mask)} objects")
            else:
                if not isinstance(value, list):
                    value = [value]
                mask = cur_metadata[key].isin(value)
                self.log.info(f"Subsetting to {key}={value}: {sum(mask)} objects")
            selected_obj_ids = selected_obj_ids[mask]
        # select ids based on nona conditions
        if nona_condition:  # TODO test this!
            if self.conditions is not None:
                mask = ~np.isnan(self.conditions).any(axis=1)
                mpp_df = self.metadata.set_index(self.data_config.OBJ_ID).loc[self.obj_ids][mask]
                nona_obj_ids = mpp_df.reset_index().groupby(self.data_config.OBJ_ID).first().index
                selected_obj_ids = np.array(list(set(selected_obj_ids).intersection(nona_obj_ids)))
                self.log.info(f"Subsetting to objects with NO_NAN condition: {len(selected_obj_ids)}")
        # select ids based on ramdom subset
        if frac is not None:
            num = int(len(selected_obj_ids) * frac)
        if num is not None:
            self.log.info(f"Subsetting to {num} random objects")
            self.rng.seed(self.seed)  # TODO: remove seeding if have duplicated existing datasets
            selected_obj_ids = selected_obj_ids[self.rng.choice(len(selected_obj_ids), size=num, replace=False)]
        # create mpp mask for applying subset
        obj_mask = self.metadata[self.data_config.OBJ_ID].isin(selected_obj_ids)
        mpp_mask = self._get_per_mpp_value(obj_mask)
        # apply mask
        return self.apply_mask(mpp_mask, copy=copy)

    def subset_channels(self, channels: Sequence[str]) -> None:
        """
        Restrict :attr:`MPPData.mpp` to ``channels``.

        Parameters
        ----------
        channels
            Channels that should be retained in mpp_data. Given as string values.

        Returns
        -------
        Nothing, updates :attr:`MPPData.mpp` and :attr:`MPPdata.channels`.
        """
        self.log.info(f"Subsetting from {len(self.channels)} channels")
        assert len(np.intersect1d(self.channels.name.values, channels)) == len(
            channels
        ), "mpp object does not contain provided channels!"
        cids = list(self.channels.reset_index().set_index("name").loc[channels]["channel_id"])
        raw_channels = self.channels
        self.channels = self.channels.loc[cids].reset_index(drop=True)
        self.channels.index.name = "channel_id"
        self._data["mpp"] = self.mpp[:, :, :, cids]
        subset_mpp_channels = self.channels
        self.log.info(f"Restricted channels to {len(self.channels)} channels")
        channels_diff = np.setdiff1d(raw_channels, subset_mpp_channels)
        if len(channels_diff) > 0:
            self.log.info(f"The following channels were excluded {channels_diff}")
        else:
            self.log.info("None of the channels were excluded ")

    def subsample(
        self,
        frac: Optional[float] = None,
        frac_per_obj: Optional[float] = None,
        num: Optional[int] = None,
        num_per_obj: Optional[int] = None,
        add_neighborhood: bool = False,
        neighborhood_size: int = 3,
    ) -> "MPPData":
        """
        Pixel-level subsampling of MPPData.

        All other information is updated accordingly (to save RAM/HDD-memory).
        Additionally, can extend mpps' representations by their neighbourhoods before subsampling.

        Note that at least one of four parameters that indicate subsampling size
        (`frac`, `num`, `frac_per_obj`, `num_per_obj`) should be provided.

        Parameters
        ----------
        frac
            Subsample a random number of mpps (pixels) from the whole dataset by a specified fraction.
            Should be in range [0, 1].
        num
            Subsample a random number of mpps (pixels) from the whole dataset by a specified number of mpps
            to be chosen.
        frac_per_obj
            Allows to subsample a fraction of mpps on the object level - that is,
            for each object (cell) to subsample
            the same fraction of mpps independently.
        num_per_obj
            Same as ``frac_per_obj``, but a number of mpps to be left instead of fraction is provided.
        add_neighborhood
            If set to True, extends mpp representation with a square neighbourhood around it.
        neighborhood_size
            Size of the neighbourhood.

        Returns
        -------
            Subsampled MPPData. Operation cannot be done in place
        """
        assert (
            sum(
                [
                    frac is not None,
                    frac_per_obj is not None,
                    num is not None,
                    num_per_obj is not None,
                ]
            )
            == 1
        ), "set only one of the params to a value"
        assert not (self.has_neighbor_data and add_neighborhood), "cannot add neighborhood, already has neighbor data"
        if (frac is not None) or (num is not None):
            if frac is not None:
                num = int(len(self.mpp) * frac)
            self.log.info(f"Subsampling from {len(self.mpp)} to {num}")
            # randomly select num mpps
            # TODO: replace with self.rng as soon as have reproduced data
            rng = np.random.default_rng(seed=self.seed)
            selected = rng.choice(len(self.mpp), num, replace=False)
            # select other information accordingly
            mpp_data = self.apply_mask(selected, copy=True)

        else:  # frac_per_obj or num_per_obj are specified
            if frac_per_obj is not None:
                self.log.info(f"Subsampling each object to {frac_per_obj*100}%")
            else:  # num_per_obj is specified
                self.log.info(f"Subsampling each object to {num_per_obj}")
            # iterate over all obj_ids
            mask = np.zeros(len(self.mpp), dtype=bool)
            idx = np.arange(len(mask))
            for obj_id in self.unique_obj_ids:
                obj_mask = self.obj_ids == obj_id
                if frac_per_obj is not None:
                    cur_num = int(obj_mask.sum() * frac_per_obj)
                else:  # num_per_obj is specified
                    cur_num = num_per_obj  # type: ignore[assignment]
                # TODO replace with self.rng if have reproduced datasets
                rng = np.random.default_rng(seed=self.seed)
                selected = rng.choice(len(obj_mask.nonzero()[0]), cur_num, replace=False)
                selected_idx = idx[obj_mask][selected]
                mask[selected_idx] = True
            mpp_data: MPPData = self.apply_mask(mask, copy=True)  # type: ignore[no-redef]
        if add_neighborhood:
            self.log.info(f"Adding neighborhood of size {neighborhood_size}")
            neighbor_mpp = self._get_neighborhood(mpp_data.obj_ids, mpp_data.x, mpp_data.y, size=neighborhood_size)
            mpp_data._data["mpp"] = neighbor_mpp
        return mpp_data

    def add_neighborhood(self, size: int = 3, copy: bool = False) -> "MPPData":
        """
        Add a square neighbourhood around each pixels to the mpp_data.

        Parameters
        ----------
        neighborhood_size
            Size n of the neighbourhood.
            The resulting mpp representation will then be set
            to a `n x n` square around each pixel.
        copy
            return new MPPData object or modify in place.

        Returns
        -------
            The modified MPPData.
        """
        self.log.info(f"Adding neighborhood of size {size}")
        mpp = self._get_neighborhood(self.obj_ids, self.x, self.y, size=size)
        assert (mpp[:, size // 2, size // 2, :] == self.center_mpp).all()

        if copy:
            data = self._data.copy()
            data["mpp"] = mpp
            return MPPData(
                metadata=self.metadata,
                channels=self.channels,
                data=data,
                seed=self.seed,
                data_config=self.data_config_name,
            )
        else:
            assert not self.has_neighbor_data, "cannot add neighborhood, already has neighbor data"
            self._data["mpp"] = mpp
            return self

    def normalise(
        self,
        background_value: Optional[Union[float, List[float], str]] = None,
        percentile: Optional[float] = None,
        rescale_values: Optional[List[float]] = None,
    ) -> None:
        """
        Normalise :attr:`MPPData.mpp` values.

        This operation is performed in place.

        Parameters
        ----------
        background_value
            Shift all values by that value. All resulting negative values are cut off at zero.
            Several shift options are possible, depending on a background_value,
            which could be in one of the following formats:

            - single value (float): then data is shifted by this number in every channels
            - list of predefined values (list of float): data in each channel is shifted separately by a corresponding
              value in that list. Note: in that case, number of shifted
              values should be the same as number of channels.
            - string: list of values for shifting are loaded from the channels_metadata table
              (should be located in ``data_config.DATA_DIR`` in a file ``data_config.CHANNELS_METADATA)``.
              The values are then loaded from the column corresponding to a provided string value.

        percentile
            Rescale the data using the specified percentile. Modifies ``rescale_values`` as a side effect with
            the calculated percentiles.
        rescale_values
            Can optionally contain previously calculated percentiles that will be used to rescale the data, instead
            of calculating percentiles again.
            This is useful for using the same values to scale e.g. train and test sets.
            If an empty list is passed, this will be filled with the percentiles calculated on this data.

        Returns
        -------
        Nothing, updates `MPPData.mpp` in place.
        """
        if background_value is not None:
            self._subtract_background(background_value)
        if percentile is not None:
            self._rescale_intensities_per_channel(percentile, rescale_values)

    # --- Helper functions ---
    def apply_mask(self, mask: np.ndarray, copy: bool = False) -> "MPPData":
        """
        Return new MPPData by applying mask to :meth:`MPPData.data`.

        Parameters
        ----------
        mask
            Boolean mask.
        copy
            Return new object.

        Returns
        -------
        Masked object.
        """
        data = {}
        for key in self._data.keys():
            data[key] = self._data[key][mask]

        if copy is False:
            self._data = data
            self.metadata = self.metadata[self.metadata[self.data_config.OBJ_ID].isin(np.unique(self.obj_ids))]
            return self
        else:
            return MPPData(
                metadata=self.metadata,
                channels=self.channels,
                data=data,
                seed=self.seed,
                data_config=self.data_config_name,
            )

    def _subtract_background(self, background_value: Union[float, List[float], str]) -> None:
        """
        Subtract background value.

        Code copied/adapted from Scott Berry's MCU package.

        Parameters
        ----------
        background_value
            Float value to be subtracted, or name of column in CHANNELS_METADATA.
        """
        if isinstance(background_value, float):
            self.log.info(f"Subtracting constant value {background_value} from all channels")
            self._data["mpp"] = self.mpp.astype(np.float32) - background_value
        elif isinstance(background_value, list):
            assert len(background_value) == len(self.channels), "specify one background value per channel"
            self.log.info("Subtracting predefined channel specific background value")
            self._data["mpp"] = self.mpp.astype(np.float32) - np.array(background_value)
        else:  # is column name
            self.log.info(f"Subtracting channel-specific background value defined in column {background_value}")
            channels_metadata = pd.read_csv(os.path.join(self.data_config.DATA_DIR, self.data_config.CHANNELS_METADATA))
            background_value_df = channels_metadata.set_index("name").loc[self.channels.name][background_value]
            # replace nan with 0
            for ch, val in zip(self.channels.name, background_value_df):
                if pd.isna(val):
                    self.log.warning(f"Missing background value for channel {ch}")
            background_value_df = background_value_df.fillna(0)
            self.log.debug(f"use background_value: {background_value_df}")
            self._data["mpp"] = self.mpp.astype(np.float32) - np.array(background_value_df)
        # cut off at 0 (no negative values)
        self._data["mpp"][self.mpp < 0] = 0

    def _rescale_intensities_per_channel(
        self, percentile: float = 98.0, rescale_values: Optional[List[float]] = None
    ) -> None:
        """
        Normalise intensity values.

        Calculates percentile and returns values/percentile.
        Uses percentile in ``rescale_values`` if given.
        Otherwise modifies ``rescale_values`` as side-effect.

        NOTE converts self.mpp to float32
        """
        if rescale_values is None:
            rescale_values = []
        if len(rescale_values) == 0:
            self.log.info(f"Rescaling MPP intensities per channel with {percentile} percentile")
            rescale_values.extend(np.percentile(self.center_mpp, percentile, axis=0))
        else:
            assert len(rescale_values) == len(
                self.channels
            ), f"need {len(self.channels)} rescale values, got {len(rescale_values)}"
            self.log.info("Rescaling MPP intensities per channels with predefined values")
        self._data["mpp"] = self.mpp / rescale_values
        self._data["mpp"] = self.mpp.astype(np.float32)

    # ---- getting functions -----
    def _get_per_mpp_value(
        self, per_obj_value: Union[Mapping[str, Iterable[Any]], Iterable[Any]]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Propagate per object values to all pixels.

        Takes list of values corresponding to self.metadata[OBJ_ID]
        and propagates them to self.obj_id.

        Parameters
        ----------
        per_obj_value
            List of values, or dict containing multiple value lists.

        Returns
        -------
        pd.Series or pd.DataFrame.
        """
        # val: Union[List[str], str] = None
        if isinstance(per_obj_value, dict):
            val: Union[List[str], str] = list(per_obj_value.keys())
            per_obj_value.update({"MERGE_KEY": self.metadata[self.data_config.OBJ_ID]})
            per_obj_df = pd.DataFrame(per_obj_value)
        else:
            val = "val"
            per_obj_df = pd.DataFrame(
                {
                    "val": per_obj_value,
                    "MERGE_KEY": self.metadata[self.data_config.OBJ_ID],
                }
            )
        df = pd.DataFrame({"MERGE_KEY": self.obj_ids})
        per_mpp_value = df.merge(per_obj_df, left_on="MERGE_KEY", right_on="MERGE_KEY", how="left")[val]
        return per_mpp_value

    def get_condition(
        self, desc: Union[List[str], str], cond_params: Optional[MutableMapping[str, Any]] = None
    ) -> np.ndarray:
        """
        Get condition based on ``desc``.

        For complete description of ``desc``, see :meth:`MPPData.add_conditions`.

        If ``desc`` is a list of conditions, return unique one-hot encoded vector combining multiple conditions
        (only possible when all sub-conditions are one-hot encoded)

        ``cond_params`` are modified in place.

        Parameters
        ----------
        cond_desc
            Conditions to be added.
        cond_params
            Can optionally contain precomputed quantiles or mean/string values. If no values are provided,
            this will be filled with computed quantiles or mean/string values.
            This is useful for using the same values to process conditions on e.g. train and test sets.

        Returns
        -------
        condition vector
        """
        if cond_params is None:
            cond_params = {}
        cond = None
        if isinstance(desc, list):
            # check if combining is possible
            assert np.all([("one_hot" in d) for d in desc]), f"all of {desc} need to be one_hot encoded"
            conds = [self.get_condition(d, cond_params) for d in desc]
            cond = get_combined_one_hot(conds)
        else:
            # desc is one column in metadata
            # get correct desc name (match to column names)
            desc, postprocess = process_condition_desc(desc)
            self.log.info(f"Looking up condition {desc}, with postprocess {postprocess}")
            cond = self._get_per_mpp_value(np.array(self.metadata[desc]))
            if postprocess == "one_hot":
                cond = convert_condition(
                    np.array(cond),
                    desc=desc,
                    one_hot=True,
                    data_config=self.data_config,
                )
            elif postprocess == "bin_3":
                cond, _ = get_bin_3_condition(cond, desc, cond_params)
                cond = get_one_hot(cond, nb_classes=3)
            elif postprocess == "lowhigh_bin_2":
                cond_bin, q = get_lowhigh_bin_2_condition(cond, desc, cond_params)
                cond_one_hot = get_one_hot(cond_bin, nb_classes=2)
                cond_one_hot[np.logical_and(cond > q[0], cond < q[1])] = np.nan
                cond = cond_one_hot
            elif postprocess == "zscore":
                cond, _ = get_zscore_condition(cond, desc, cond_params)
                # add empty axis for concatenation
                cond = np.array(cond[:, np.newaxis])
            else:
                cond = convert_condition(
                    np.array(cond)[:, np.newaxis],
                    desc=desc,
                    data_config=self.data_config,
                )
        return cond

    def get_adata(
        self, X: str = "mpp", obsm: Union[Dict[str, str], List[str], Tuple[str, ...]] = (), obs: Iterable[str] = ()
    ) -> ad.AnnData:
        """
        Create adata from information contained in MPPData.

        Channels are put in ``adata.var``.
        Metadata is put in ``adata.obs``.

        Parameters
        ----------
        X
            Key in MPPData.data that should be in adata.X.
        obsm
            Keys in MPPData.data that should be in adata.obsm.
            Can be dict with keys the desired names in adata, and values the keys in MPPData.data
        obs
            Keys from MPPData.data that should be in adata.obs in addition to all information from metadata.

        Returns
        -------
        adata with pixel-level observations.
        """
        import anndata as ad

        if isinstance(obsm, list) or isinstance(obsm, tuple):
            obsm = {o: o for o in obsm}

        assert len(set(obsm.values()).intersection(self._data.keys())) == len(obsm)
        obsm = {k: self.data(v).astype(np.float32) for k, v in obsm.items()}  # type: ignore[union-attr]
        if X == "mpp":
            var = self.channels
            X_ = self.center_mpp
        else:
            var = None
            X_ = self._data[X]
        # add spatial coords as obsm['spatial']
        obsm["spatial"] = np.stack([self.x, self.y]).T.astype(np.float32)
        # get per-pixel obs
        obs_ = self._get_per_mpp_value(self.metadata.to_dict(orient="list"))
        for o in obs:
            obs_[o] = self.data(o)
        adata = ad.AnnData(X=X_.astype(np.float32), obs=obs_, var=var, obsm=obsm)
        # set var_names if possible
        if var is not None:
            adata.var_names = adata.var["name"]
        return adata

    def extract_csv(self, data: str = "mpp", obs: Iterable[str] = ()) -> pd.DataFrame:
        """
        Extract information in :meth:`MPPData.data` into :class:`pd.DataFrame`.

        Per default, files contain ``data``, ``x`` and ``y`` coordinates, and ``obj_id``.

        Parameters
        ----------
        obs
            list of :attr:`MPPData.metadata` columns that should be additionally added.

        Returns
        -------
        DataFrame containing mpp data.
        """
        # TODO: column names are tuples
        columns = None
        if data == "mpp":
            columns = self.channels["name"]
            X = self.center_mpp
        else:
            X = self._data[data]
        df = pd.DataFrame(data=X, columns=columns)
        # add x,y,obj_id
        df["x"] = self.x
        df["y"] = self.y
        obs_ = self._get_per_mpp_value({o: self.metadata[o] for o in list(obs) + [self.data_config.OBJ_ID]})
        for key, value in obs_.items():
            df[key] = value
        return df

    def _get_neighborhood(
        self,
        obj_ids: Union[np.ndarray, List[str]],
        xs: np.ndarray,
        ys: np.ndarray,
        size: int = 3,
        border_mode: str = "center",
    ) -> np.ndarray:
        """Return neighbourhood information for given obj_ids + xs + ys."""
        data = np.zeros((len(obj_ids), size, size, len(self.channels)), dtype=self.mpp.dtype)
        for obj_id in np.unique(obj_ids):
            mask = obj_ids == obj_id
            img, (xoff, yoff) = self.get_object_img(obj_id, data="mpp", pad=size // 2)
            vals = []
            for x, y in zip(xs[mask], ys[mask]):
                idx = tuple(slice(pp - size // 2, pp + size // 2 + 1) for pp in [y - yoff, x - xoff])
                vals.append(pad_border(img[idx], mode=border_mode))
            data[mask] = np.array(vals)
        return data

    def get_channel_ids(
        self, to_channels: List[str], from_channels: Optional[Union[pd.DataFrame, List[str]]] = None
    ) -> List[int]:
        """
        Map channel names to ids.

        Parameters
        ----------
        to_channels
            Channel names for which would like to know the ids.
        from_channels
            Get ``channel_ids`` assuming ordering of ``from_channels``. By default use the ordering
            of :attr:`MPPData.channels`.
            This is useful for models that are trained with different output than input channels.

        Returns
        -------
        Channel ids that can be used to index :attr:`MPPData.mpp`.
        """
        if from_channels is None:
            from_channels = self.channels.copy()
            from_channels = from_channels.reset_index().set_index("name")  # type: ignore[union-attr]
        if not isinstance(from_channels, pd.DataFrame):
            from_channels = pd.DataFrame({"name": from_channels, "channel_id": np.arange(len(from_channels))})
            from_channels = from_channels.set_index("name")
        from_channels = from_channels.reindex(to_channels)
        return list(from_channels["channel_id"])

    # --- image plotting (only for non-subsampled MPPData) ---
    @staticmethod
    def _get_img_from_data(
        x: np.ndarray, y: np.ndarray, data: np.ndarray, img_size: Optional[int] = None, pad: int = 0
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Create image from x and y coordinates and fill with data.

        Args:
            x, y: 1-d array of x and y corrdinates
            data: shape (n_coords,n_channels), data for the image
            img_size: size of returned image
            pad: amount of padding added to returned image (only used when img_size is None)
        """
        x_coord = x - x.min() + pad
        y_coord = y - y.min() + pad
        # create image
        img = np.zeros(
            (y_coord.max() + 1 + pad, x_coord.max() + 1 + pad, data.shape[-1]),
            dtype=data.dtype,
        )
        img[y_coord, x_coord] = data
        # resize
        if img_size is not None:
            c = BoundingBox(0, 0, img.shape[0], img.shape[1]).center
            bbox = BoundingBox.from_center(c[0], c[1], img_size, img_size)
            img = bbox.crop(img)
            return img
        else:
            # offset info is only relevant when not cropping img to shape
            return img, (int(x.min() - pad), int(y.min() - pad))

    def get_object_img(
        self,
        obj_id: str,
        data: str = "mpp",
        channel_ids: Optional[Iterable[int]] = None,
        annotation_kwargs: Mapping[str, Any] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Calculate data image of given object id.

        Parameters
        ----------
        obj_id
            Object that should be visualised.
        data
            Key in :meth:`MPPData.data` that should be plotted on the image.
        channel_ids
            Only if key == 'mpp'. Channels that the image should have.
            If None, all channels are returned.
        annotation_kwargs
            Arguments for :func:`campa.pl.annotate_img` (``annotation``, ``to_col``, ``color``).
            Use this to plot clusterings on cells with custom colormaps.
        img_size
            Size of returned image. If None, images have minimal required size to fit all information.
            If None, offset information is returned by this function.
        pad
            Amount of padding added to returned image (only used when ``img_size`` is None).

        Returns
        -------
        If img_size is None, returns: ``image, (offset_x, offset_y)``.
            Offset is that value that needs to be subtracted from :attr:`MPPData.x` and :attr:`MPPData.y`
            before being able to use them to index the returned image.
        Otherwise returns: image

        """
        mask = self.obj_ids == obj_id
        x = self.x[mask]
        y = self.y[mask]
        if data == "mpp":
            values = self.center_mpp[mask]
            if channel_ids is not None:
                # print('channel_ids:', channel_ids, type(channel_ids))
                # print('values:', values, values.shape, values.dtype)
                values = values[:, channel_ids]
        else:
            values = self._data[data][mask]
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        # color image if necessary
        img = MPPData._get_img_from_data(x, y, values, **kwargs)
        if annotation_kwargs is not None:
            if isinstance(img, tuple):  # have padding info in img
                img_ = annotate_img(img[0], from_col=data, **annotation_kwargs)
                img = (img_, img[1])
            else:
                img = annotate_img(img, from_col=data, **annotation_kwargs)
        return img

    def get_object_imgs(
        self,
        data: str = "mpp",
        channel_ids: Optional[Iterable[int]] = None,
        annotation_kwargs: Mapping[str, Any] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """
        Return images for each obj_id in current data.

        Parameters
        ----------
        data
            Key in :meth:`MPPData.data` that should be plotted on the image.
        channel_ids
            Only if key == 'mpp'. Channels that the image should have.
            If None, all channels are returned.
        annotation_kwargs
            Arguments for :func:`campa.pl.annotate_img` (``annotation``, ``to_col``, ``color``).
            Use this to plot clusterings on cells with custom colormaps.
        img_size
            Size of returned image. If None, images have minimal required size to fit all information.
            If None, offset information is returned by this function.
        pad
            Amount of padding added to returned image (only used when ``img_size`` is None).

        Returns
        -------
        list of images.
        """
        imgs = []
        for obj_id in self.metadata[self.data_config.OBJ_ID]:
            res = self.get_object_img(
                obj_id=obj_id,
                data=data,
                channel_ids=channel_ids,
                annotation_kwargs=annotation_kwargs,
                **kwargs,
            )
            if kwargs.get("img_size", None) is None:
                # fn also returns padding info, which we don't need here
                res = res[0]
            imgs.append(res)
        return imgs  # type: ignore[return-value]

    def get_img(
        self, data: str = "mpp", channel_ids: Optional[Iterable[str]] = None, **kwargs: Any
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Calculate data image of entire MPPData.

        Parameters
        ----------
        data
            Key in :meth:`MPPData.data` that should be plotted on the image.
        channel_ids
            Only if key == 'mpp'. Channels that the image should have.
            If None, all channels are returned.
        img_size
            Size of returned image. If None, images have minimal required size to fit all information.
            If None, offset information is returned by this function.
        pad
            Amount of padding added to returned image (only used when ``img_size`` is None).

        Returns
        -------
        If img_size is None, returns: image, (offset_x, offset_y).
            Offset is that value that needs to be subtracted from MPPData.x and MPPData.y
            before being able to use them to index the returned image.
        Otherwise returns: image
        """
        if data == "mpp":
            values = self.center_mpp
            if channel_ids is not None:
                values = values[:, channel_ids]
        else:
            values = self._data[data]
        if len(values.shape) == 1:
            values = values[:, np.newaxis]
        return MPPData._get_img_from_data(self.x, self.y, values, **kwargs)

    def _compare(self, obj: "MPPData") -> Tuple[bool, Dict[str, bool]]:
        """
        Compare two MPPDatas and return their differences.

        Used for testing.
        """
        assert np.array_equal(list(self._data.keys()), list(obj._data.keys()))
        same_data = {}
        for key in self._data.keys():
            issame = np.array_equal(self._data[key], obj._data[key])
            same_data[key] = issame

        same_data["channels"] = self.channels.equals(obj.channels)
        if len(self.metadata) == len(obj.metadata):
            same_data["metadata"] = self.metadata.reset_index(drop=True).equals(obj.metadata.reset_index(drop=True))
        else:
            same_data["metadata"] = False
        return np.all(list(same_data.values())), same_data  # type: ignore[return-value]


def _try_mmap_load(fname, mmap_mode="r", allow_not_existing=False):
    """
    Use np.load to read fname.

    If mmap loading fails, load with mmap_mode=None.

    Args:
        allow_not_existing: if fname does not exist, do not raise and exception, just return None

    Returns:
        np.ndarray: loaded numpy array
    """
    try:
        try:
            res = np.load(fname, mmap_mode=mmap_mode)
        except ValueError as e:
            if "Array can't be memory-mapped: Python objects in dtype." in str(e):
                print("Cannot read with memmap: ", fname)
                res = np.load(fname, mmap_mode=None, allow_pickle=True)
            else:
                raise e
    except FileNotFoundError as e:
        if allow_not_existing:
            return None
        else:
            raise e
    return res


def _get_keys(keys, optional_keys, base_mpp_data=None):
    """
    Return keys and optional keys to use for loading base_mpp_data / mpp_data.

    Used when reading parts of the data from other dirs.
    """
    if base_mpp_data is None:
        # loading of base_mpp_data
        # return all keys as optional, as might still be present in mpp_data_dir
        res_keys = ["x", "y", "obj_ids"]
        res_optional_keys = list(set(optional_keys).union(keys).difference(res_keys))
    else:
        # loading of mpp_data
        # check which of required keys are already loaded, and move them to optional
        res_keys = ["x", "y", "obj_ids"]
        res_optional_keys = list(copy(optional_keys))
        for k in list(set(keys).difference(res_keys)):
            if base_mpp_data.data(k) is not None:
                res_optional_keys.append(k)
            else:
                res_keys.append(k)
    return res_keys, res_optional_keys
