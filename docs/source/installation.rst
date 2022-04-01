Installation
------------

Campa is developed for and tested on Python 3.9.

Installation with pip
=====================
::

    git clone https://github.com/theislab/campa
    cd campa
    pip install .


Setup
=====

To find data and experiment folders, and dataset specific parameters,
CAMPA needs a config file, ``campa.ini``.
This file contains the following sections:

- ``[DEFAULT]``
    - ``experiment_dir``: default path to the directory where experiment
      results are stored.
      This is going to be loaded in ``campa.constants.EXPERIMENT_DIR``
    - ``data_dir``: default path of the directory where all data is stored.
      For more information on the data format, see the `MPPData notebook`_.
      Note that any given data config for specific data may overwrite
      ``data_dir``, and all functions will use ``data_config.DATA_DIR``
      to determine the path to the data.
      ``campa.constants.BASE_DATA_DIR`` is useful to set a system-specific root
      for data folders, allowing to re-use data configs accross systems.
- ``[data]``
    - ``DataName=<path>``. Each constant here contains a path to a python file,
      where parameters that are specific to the required data load are stored.
      Please refer to the `MPPData notebook`_ and
      the `ExampleData_constants.py <https://github.com/theislab/campa/blob/main/notebooks/params/ExampleData_constants.py>`_
      to see an example.

.. _MPPData notebook: notebooks/mpp_data.ipynb

CAMPA will look for config files in:

- the CAMPA code directory
- the current directory
- ``$HOME/.config/campa``

and will use the first file that it finds.

There is an example config file in ``campa/campa.ini.example``.
Create ``campa.ini`` with system-specific paths to experiment and
data folders in by running::

    campa setup

This creates a default config in ``$HOME/.config/campa/campa.ini``.
To add new ``[data]`` fields, directly edit ``campa.ini``
in ``$HOME/.config/campa``.
