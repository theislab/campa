# read constants defined in config.ini
from configparser import ConfigParser, NoOptionError, NoSectionError
from campa.utils import load_config
import os

def get_value(config, key, section="DEFAULT"):
    try:
        return config.get(section, key)
    except (NoSectionError, NoOptionError) as e:
        print(f'WARNING: {key} not defined in {config_fname}')
        return None

# scripts dir is parent dir of this file (campa folder)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# expect config in scripts dir
config_fname = os.path.abspath(os.path.join(SCRIPTS_DIR, 'config.ini'))
config = ConfigParser()
config.read(config_fname)

# get paths defined in config
EXPERIMENT_DIR = get_value(config, key='experiment_dir')
BASE_DATA_DIR = get_value(config, key='data_dir')

def get_data_config(data_config="NascentRNA"):
    return load_config(get_value(config, section='data', key=data_config))

CO_OCC_CHUNK_SIZE = float(get_value(config, key='co_occ_chunk_size', section='co_occ'))

# enums
from enum import Enum

class CoOccAlgo(Enum):
    SQUIDPY = 'squidpy'
    OPT = 'opt'

