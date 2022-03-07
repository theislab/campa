# CAMPA

Conditional Autoencoder for Muliplexed Pixel Analysis

## Installation

Campa is developed for and tested on Python 3.9. 

### Installation with pip

    git clone https://github.com/theislab/campa
    cd campa
    pip install .


### Setup

Create `campa.ini` with system-specific paths to experiment and data folders by running 
```
python cli/setup.py
```
This creates a default config in `$HOME/.config/campa/campa.ini`. 
This config contains the following sections
- `[DEFAULT]`
    - `experiment_dir`: default path to the directory where experiment results are stored. This is going to be loaded in `campa.constants.EXPERIMENT_DIR`
    - `data_dir`: default path of the directory where all data is stored. For more information on the data format, see the [MPPData notebook](notebooks/mpp_data.ipynb) and the [example_data](notebooks/example_data) folder. 
    Note that any given data config for specific data may overwrite `data_dir`, and all functions will use `data_config.DATA_DIR` to determine the path to the data. `campa.constants.BASE_DATA_DIR` is useful to set a system-specific root for data folders, allowing to re-use data configs accross systems.
- `[data]`
    - `DataName=<path>`. Each constant here contains a path to a python file, where parameters that are specific to the required data load are stored. Please refer to the [MPPData notebook](notebooks/mpp_data.ipynb) and the [TestData_constants.py](notebooks/TestData_constants.py) to see an example.
To add new `[data]` fields, directly edit `campa.ini` in `$HOME/.config/campa`. Note that CAMPA will look for `campa.ini` in the current directory as well as in `$HOME/.config/campa`.


# TODO

- add test + cluster + eval notebooks to test
- write feature notebook
- write notebook explaining how to export things
- create a list of tests that still need to be done
- incorporate readthedocs in tox
- add black + import checking to precommit + set up precommit