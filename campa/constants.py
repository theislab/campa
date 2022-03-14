# read constants defined in config.ini
from configparser import ConfigParser, NoOptionError, NoSectionError
from campa.utils import load_config
import os
from pathlib import Path

def get_value(config, key, section="DEFAULT"):
    try:
        return config.get(section, key)
    except (NoSectionError, NoOptionError) as e:
        print(f'{key} not defined in {config_fnames}')
        return None

# scripts dir is parent dir of this file (campa folder)
SCRIPTS_DIR = Path(__file__).parent.parent

# look for configs in .config folder and in current dir
config_fnames = [
    Path.home() / '.config' / 'campa' / 'campa.ini',
    Path.cwd() / 'campa.ini',
    Path(__file__).resolve().parent.parent / 'config.ini' # TODO this is for legacy reasons, should be removed at some point
]
if sum([n.is_file() for n in config_fnames]) == 0:
    print(f'None of {config_fnames} exists. Please create a config with "python cli/setup.py"')
print(f"Reading config from {config_fnames}")

config = ConfigParser()
config.read(config_fnames)

# get paths defined in config
EXPERIMENT_DIR = get_value(config, key='experiment_dir')
BASE_DATA_DIR = get_value(config, key='data_dir')

def get_data_config(data_config="TestData"):
    return load_config(get_value(config, section='data', key=data_config))

CO_OCC_CHUNK_SIZE = get_value(config, key='co_occ_chunk_size', section='co_occ')
if CO_OCC_CHUNK_SIZE is not None:
    CO_OCC_CHUNK_SIZE = float(CO_OCC_CHUNK_SIZE)

# enums
from enum import Enum

class CoOccAlgo(Enum):
    SQUIDPY = 'squidpy'
    OPT = 'opt'

