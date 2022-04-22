from copy import deepcopy
from typing import Any, Mapping, MutableMapping
import os
import logging
import collections.abc


def init_logging(level=logging.INFO):
    import warnings

    # from numba.errors import NumbaPerformanceWarning
    from anndata import ImplicitModificationWarning

    logging.basicConfig(level=level)  # need one of this?
    logging.getLogger().setLevel(level)  # need one of this?
    # ignore tensorflow warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    # ignore scanpy / anndata warnings
    logging.getLogger("scanpy").setLevel(logging.WARNING)
    logging.getLogger("anndata").setLevel(logging.ERROR)
    # logging.getLogger('get_version').setLevel(logging.WARNING)
    # logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
    # ignore numba warnings
    # warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    # filter irrelevant / non-fixeable anndata warnings
    # anndata uses inplace from pandas, which is depreciated
    warnings.filterwarnings("ignore", category=FutureWarning, 
        message='.*The `inplace` parameter in pandas.Categorical.reorder_categories \
is deprecated and will be removed in a future version. \
Removing unused categories will always return a new Categorical object.*')
    # implicit modification warnings from anndata
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)


def load_config(config_file):
    """load config file and return config object"""
    import importlib.util
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(os.path.basename(config_file), config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is not None:
        config = importlib.util.module_from_spec(spec)
        loader.exec_module(config)
        return config
    return None


def merged_config(config1: MutableMapping[str, Any], config2: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """update config1 with config2. Should work arbitary nested levels"""
    res_config = deepcopy(config1)
    for k, v in config2.items():
        if isinstance(v, collections.abc.Mapping):
            res_config[k] = merged_config(config1.get(k, {}), v)
        else:
            res_config[k] = v
    return res_config
