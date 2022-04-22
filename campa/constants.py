# read constants defined in config.ini
from enum import Enum
from typing import Any, Union, Optional
from pathlib import Path
from configparser import ConfigParser, NoOptionError, NoSectionError
import os
import multiprocessing

from campa.utils import load_config


def _get_value(config: ConfigParser, key: str, section: str = "DEFAULT") -> Optional[str]:
    try:
        return config.get(section, key)
    except (NoSectionError, NoOptionError):
        print(f"{key} not defined in config file")
        return None


class CAMPAConfig:
    """
    Config for CAMPA.

    Defines device-specific EXPERIENT_DIR and BASE_DATA_DIR paths and
    mapping of data config names to config files.

    The config is attempted to be read from a "campa.ini" config file.
    This file is searched for in the following places:
    - the current directory
    - ``$HOME/.config/campa/``
    A default config file can be created using ``campa setup``

    Config values can also directly be set in memory.
    """

    def __init__(self):
        self._EXPERIMENT_DIR = None
        self._BASE_DATA_DIR = None
        self._data_configs = {}
        self._CO_OCC_CHUNK_SIZE = None
        self._config_fname = self._discover_config_file()
        if self._config_fname is not None:
            self._load_config_from_file()

    def _load_config_from_file(self):
        assert self.config_fname is not None
        config = ConfigParser()
        if multiprocessing.parent_process() is None:  # silence print in child processes
            print(f"Reading config from {str(self.config_fname)}")
        config.read(self.config_fname)

        self._EXPERIMENT_DIR = _get_value(config, key="experiment_dir")
        self._BASE_DATA_DIR = _get_value(config, key="data_dir")
        data_configs = dict(config.items("data"))
        data_configs.pop("experiment_dir")
        data_configs.pop("data_dir")
        self._data_configs = data_configs
        self._CO_OCC_CHUNK_SIZE = _get_value(config, key="co_occ_chunk_size", section="co_occ")

    def _discover_config_file(self) -> Optional[Union[Path, str]]:
        # look for configs in .config folder and in current dir
        config_fnames = [
            Path.cwd() / "campa.ini",
            Path(__file__).resolve().parent.parent
            / "config.ini",  # TODO this is for legacy reasons, should be removed at some point
            Path.home() / ".config" / "campa" / "campa.ini",
        ]
        for n in config_fnames:
            if n.is_file():
                return n
        return None

    def __str__(self):
        s = f"CAMPAConfig (fname: {self.config_fname})"
        s += f"\nEXPERIMENT_DIR: {self.EXPERIMENT_DIR}\nBASE_DATA_DIR: {self.BASE_DATA_DIR}\n"
        s += f"CO_OCC_CHUNK_SIZE: {self.CO_OCC_CHUNK_SIZE}\n"
        for name, val in self.data_configs.items():
            s += f"data_config/{name}: {val}\n"
        return s

    @property
    def config_fname(self) -> Optional[Union[os.PathLike[str], str]]:
        """
        Name of config file.

        When changed, the config will be re-read from this file.
        """
        return self._config_fname

    @config_fname.setter
    def config_fname(self, fname: Union[os.PathLike[str], str]) -> None:
        fname = Path(fname)
        if fname.is_file():
            try:
                self._config_fname = fname
                self._load_config_from_file()
            except ValueError as e:
                raise (e)

    @property
    def EXPERIMENT_DIR(self):
        """
        Root EXPERIMENT_DIR.

        All experiments and other paths relating to experiments will be relative to EXPERIMENT_DIR.
        """
        if self._EXPERIMENT_DIR is None:
            s = "WARNING: EXPERIMENT_DIR is not initialised."
            s += ' Please create a config with "campa setup" or set campa_config.EXPERIMENT_DIR manually.'
            print(s)
        return self._EXPERIMENT_DIR

    @EXPERIMENT_DIR.setter
    def EXPERIMENT_DIR(self, experiment_dir):
        self._EXPERIMENT_DIR = os.path.abspath(experiment_dir)

    @property
    def BASE_DATA_DIR(self):
        """
        Root BASE_DATA_DIR.

        Can be used in a data_config file to set a DATA_DIR, which will be used for all paths relating to data.
        """
        if self._BASE_DATA_DIR is None:
            s = "WARNING: BASE_DATA_DIR is not initialised."
            s += ' Please create a config with "campa setup" or set campa_config.BASE_DATA_DIR manually.'
            print(s)
        return self._BASE_DATA_DIR

    @BASE_DATA_DIR.setter
    def BASE_DATA_DIR(self, data_dir):
        self._BASE_DATA_DIR = os.path.abspath(data_dir)

    @property
    def CO_OCC_CHUNK_SIZE(self):
        """
        Chunk size for optimised co_occurrence in :meth:`FeatureExtractor.extract_co_occurrence`.

        Maximal size of co-occ matrix.
        If number of coordinates that need to be checked is larger, co-occ is computed in chunks.
        Set to lower values when using multi-processing to compute co-occurrence scores.
        """
        if self._CO_OCC_CHUNK_SIZE is not None:
            return float(self._CO_OCC_CHUNK_SIZE)
        return None

    @CO_OCC_CHUNK_SIZE.setter
    def CO_OCC_CHUNK_SIZE(self, cs):
        self._CO_OCC_CHUNK_SIZE = cs

    @property
    def data_configs(self):
        """
        Dictionary mapping from names to data_config files.
        """
        return self._data_configs

    def add_data_config(self, data_config_name: str, data_config_file: str) -> None:
        """
        Add a (name, file) mapping to :attr:`CAMPAConfig.data_configs`.

        Parameters
        ----------
        data_config_name
            Name of the data config.
        data_config_file
            Path to the data_config file.
        """
        if data_config_name.lower() in self._data_configs.keys():
            print(f"Overwriting existing data config for {data_config_name.lower()}")
        self._data_configs[data_config_name.lower()] = os.path.abspath(data_config_file)

    def get_data_config(self, data_config_name: str = "ExampleData") -> Any:
        """
        Load data_config file.

        Parameters
        ----------
        data_config_name
            Name of the data config.

        Returns
        -------
        data_config module
        """
        module = load_config(self.data_configs[data_config_name.lower()])
        if module is None:
            raise ValueError(f"Unknown data_config {data_config_name.lower()}")
        else:
            return module

    def write(self, config_fname=None):
        """
        Save current config to config_fname.

        Parameters
        ----------
        config_fname
            Name of the config file. Default is :attr:`CAMPAConfig.config_fname`.
        """
        if config_fname is None:
            if self.config_fname is None:
                raise ValueError("No config_fname specified")
            else:
                config_fname = self.config_fname
        assert self._EXPERIMENT_DIR is not None, "EXPERIMENT_DIR needs to be set."
        assert self._BASE_DATA_DIR is not None, "BASE_DATA_DIR needs to be set."
        # create configParser
        config = ConfigParser()
        config.set("DEFAULT", "experiment_dir", self.EXPERIMENT_DIR)
        config.set("DEFAULT", "data_dir", self.BASE_DATA_DIR)
        config.add_section("co_occ")
        config.set("co_occ", "co_occ_chunk_size", str(self.CO_OCC_CHUNK_SIZE))
        config.add_section("data")
        for data_config_name, data_config in self.data_configs.items():
            config.set("data", data_config_name, data_config)
        # write config
        with open(config_fname, "w") as configfile:
            config.write(configfile)
        # set config_fname
        if self.config_fname != config_fname:
            self.config_fname = config_fname


campa_config: CAMPAConfig = CAMPAConfig()
"""
Config for campa (:class:`CAMPAConfig` instance).
"""

# TODO build config class here to access and interactively change config values.
# This way can set values in notebooks for more transparency
# TODO can also have write_config method to save config.
# TODO need to replace all mentions of EXPERIMENT_DIR with config.EXPERIMENT_DIR and same for base path

# def get_value(config, key, section="DEFAULT"):
#    try:
#        return config.get(section, key)
#    except (NoSectionError, NoOptionError):
#        print(f"{key} not defined in {config_fnames}")
#        return None


# scripts dir is parent dir of this file (campa folder)
SCRIPTS_DIR = Path(__file__).parent.parent

# look for configs in .config folder and in current dir
# config_fnames = [
#    Path.cwd() / "campa.ini",
#    Path(__file__).resolve().parent.parent
#    / "config.ini",  # TODO this is for legacy reasons, should be removed at some point
#    Path.home() / ".config" / "campa" / "campa.ini",
# ]
# config = ConfigParser()
# if sum(n.is_file() for n in config_fnames) == 0:
#    print(f'ERROR: None of {config_fnames} exists. Please create a config with "python cli/setup.py"')
# for n in config_fnames:
#    if n.is_file():
#        print(f"Reading config from {str(n)}")
#        config.read(n)
#        break

# get paths defined in config
# EXPERIMENT_DIR = get_value(config, key="experiment_dir")
# BASE_DATA_DIR = get_value(config, key="data_dir")


# def get_data_config(data_config: str = "ExampleData") -> Any:
#    module = load_config(get_value(config, section="data", key=data_config))
#    if module is None:
#        raise ValueError(f"Unknown data_config {data_config}")
#    else:
#        return module


# CO_OCC_CHUNK_SIZE = get_value(config, key="co_occ_chunk_size", section="co_occ")
# if CO_OCC_CHUNK_SIZE is not None:
#    CO_OCC_CHUNK_SIZE = float(CO_OCC_CHUNK_SIZE)


# enums
class CoOccAlgo(Enum):
    SQUIDPY = "squidpy"
    OPT = "opt"
