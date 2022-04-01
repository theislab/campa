# read constants defined in config.ini
from enum import Enum
from typing import Any
from pathlib import Path
from configparser import ConfigParser, NoOptionError, NoSectionError

from campa.utils import load_config


def get_value(config, key, section="DEFAULT"):
    try:
        return config.get(section, key)
    except (NoSectionError, NoOptionError):
        print(f"{key} not defined in {config_fnames}")
        return None


# scripts dir is parent dir of this file (campa folder)
SCRIPTS_DIR = Path(__file__).parent.parent

# look for configs in .config folder and in current dir
config_fnames = [
    Path.cwd() / "campa.ini",
    Path(__file__).resolve().parent.parent
    / "config.ini",  # TODO this is for legacy reasons, should be removed at some point
    Path.home() / ".config" / "campa" / "campa.ini",
]
config = ConfigParser()
if sum(n.is_file() for n in config_fnames) == 0:
    print(f'ERROR: None of {config_fnames} exists. Please create a config with "python cli/setup.py"')
for n in config_fnames:
    if n.is_file():
        print(f"Reading config from {str(n)}")
        config.read(n)
        break

# get paths defined in config
EXPERIMENT_DIR = get_value(config, key="experiment_dir")
BASE_DATA_DIR = get_value(config, key="data_dir")


def get_data_config(data_config: str = "ExampleData") -> Any:
    module = load_config(get_value(config, section="data", key=data_config))
    if module is None:
        raise ValueError(f"Unknown data_config {data_config}")
    else:
        return module


CO_OCC_CHUNK_SIZE = get_value(config, key="co_occ_chunk_size", section="co_occ")
if CO_OCC_CHUNK_SIZE is not None:
    CO_OCC_CHUNK_SIZE = float(CO_OCC_CHUNK_SIZE)


# enums
class CoOccAlgo(Enum):
    SQUIDPY = "squidpy"
    OPT = "opt"
