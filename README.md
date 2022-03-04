# campa
Conditional Autoencoder for Muliplexed Pixel Analysis

## Installation

1. Create new conda environment:

`
conda create -n pelkmans-3.9 python=3.9
conda activate pelkmans-3.9
`

2. Install tensorflow:

`
pip install --upgrade pip
pip install tensorflow
`

3. Install other packages:

`
conda install matplotlib jupyterlab pandas tqdm
`

4. Install miann  

TODO: scanpy, squidpy
```
conda create -n pelkmans-3.9 python=3.9 tensorflow matplotlib jupyterlab pandas tqdm
conda install -c conda-forge leidenalg
pip install -e .
```

### development and test
additionally need the following packages:

pytest
pytest-cov
nbconvert
pytest dependency

### Setup
Create `config.ini` with system-specific paths to experiment and data folders:
```
cp config.ini.example config.ini
```
This config contains the following sections
- `[DEFAULT]`
    - `experiment_dir`: default path to the directory where experiment results are stored. This is going to be loaded in `campa.constants.EXPERIMENT_DIR`
    - `data_dir`: default path of the directory where all data is stored. For more information on the data format, see the [MPPData notebook](notebooks/mpp_data.ipynb) and the [example_data](notebooks/example_data) folder. 
    Note that any given data config for specific data may overwrite `data_dir`, and all functions will use `data_config.DATA_DIR` to determine the path to the data. `campa.constants.BASE_DATA_DIR` is useful to set a system-specific root for data folders, allowing to re-use data configs accross systems.
- `[data]`
    - `DataName=<path>`. Each constant here contains a path to a python file, where parameters that are specific to the required data load are stored. Please refer to the [MPPData notebook](notebooks/mpp_data.ipynb) and the [TestData_constants.py](notebooks/TestData_constants.py) to see an example.

# TODO
- add test + cluster + eval notebooks to test
- write feature notebook
- write notebook explaining how to export things
- create a list of tests that still need to be done