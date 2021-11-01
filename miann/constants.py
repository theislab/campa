# read constants defined in config.ini
from configparser import ConfigParser, NoOptionError, NoSectionError
from miann.utils import load_config
import os

def get_value(config, key, section="DEFAULT"):
    try:
        return config.get(section, key)
    except (NoSectionError, NoOptionError) as e:
        print(f'WARNING: {key} not defined in {config_fname}')
        return None

# scripts dir is parent dir of this file (miann folder)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# expect config scripts dir
config_fname = os.path.abspath(os.path.join(SCRIPTS_DIR, 'config.ini'))
config = ConfigParser()
config.read(config_fname)

# get paths defined in config
EXPERIMENT_DIR = get_value(config, key='experiment_dir')

def get_data_config(data_config="NascentRNA"):
    return load_config(get_value(config, section='data', key=data_config))

