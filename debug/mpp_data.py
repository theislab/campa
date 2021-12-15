import importlib
import sys
sys.path.append('..')

import miann.data._data
from miann.data._data import MPPData
importlib.reload(miann.data._data)

from miann.utils import init_logging
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
#init_logging()

mpp_data = miann.data._data.MPPData.from_data_dir(data_dir='184A1_unperturbed/I09', data_config='NascentRNA_mpp_data')
print(mpp_data)