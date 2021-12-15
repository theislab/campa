import importlib
import sys


import logging
logging.basicConfig(level=logging.INFO)

from miann.data import NNDataset
from miann.data import create_dataset
from miann.utils import load_config


#%%

config = load_config("params/example_data_params.py")
print(config.data_params)


#%%

create_dataset(config.data_params)

#%% md

## Use NNDataset

#%%

dataset_name = '184A1_test_dataset'
ds = NNDataset(dataset_name, data_config='NascentRNA_test')

#%%

# dataset has attributes x and y (NN input + output)
# x is either mpp or mpp+condition
x = ds.x('val', is_conditional=False)
print(x.shape)

x, c = ds.x('train', is_conditional=True)
print(x.shape, c.shape)

#%%

# dataset has data attributes with train/val/test data and img attribute with val/test image data.
# each split is represented as an MPPData object
print(ds.data['train'])

#%%

# dataset can return a tf dataset for using with e.g. keras
tf_ds = ds.get_tf_dataset(split='train', is_conditional=True)
print(tf_ds)

for x,y in tf_ds.take(1):
    print(x)
    print(y)

#%%

# dataset has fn for mapping channel orderings to channel ids TODO when do we need this?
print(ds.get_channel_ids(['16_H3', '09_CCNT1']))

#%%


