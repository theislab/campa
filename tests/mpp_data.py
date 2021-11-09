
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

mpp_data = miann.data._data.MPPData.from_data_dir('184A1_unperturbed/I11', data_config='NascentRNA_test')
print(mpp_data)

# subset channels
channels = [
        '01_CDK9_pT186', '01_PABPC1', '02_CDK7', '03_CDK9', '03_RPS6', '05_GTF2B', '05_Sm', '07_POLR2A', '07_SETD1A',
        '08_H3K4me3', '09_CCNT1', '09_SRRM2', '10_H3K27ac', '10_POL2RA_pS2', '11_KPNA2_MAX', '11_PML', '12_RB1_pS807_S811',
        '12_YAP1', '13_PABPN1', '13_POL2RA_pS5', '14_PCNA', '15_SON', '15_U2SNRNPB', '16_H3', '17_HDAC3', '17_SRSF2',
        '18_NONO', '19_KPNA1_MAX', '20_ALYREF', '20_SP100', '21_COIL', '21_NCL', '00_DAPI', '07_H2B'
        ]
print(mpp_data.channels)
mpp_data.subset_channels(channels)
print(mpp_data.channels)

# filter by NO_NAN cellcycle stage
mpp_data.subset(cell_cycle='NO_NAN')
print(np.unique(mpp_data.metadata.cell_cycle))

# get condition for each mpp
TR = mpp_data.get_condition('TR')[:,0]
print(TR)



# add conditions
cond_params = {}
mpp_data.add_conditions(['TR_norm_lowhigh_bin_2', 'TR_bin_3', 'cell_cycle'], cond_params=cond_params)

print(mpp_data.conditions)
print(cond_params)



# filter nan objects with nan conditions
mpp_data.subset(nona_condition=True)


# normalise
rescale_values = []
mpp_data.normalise(background_value='mean_background', percentile= 98.0, rescale_values=rescale_values)
print(rescale_values)


# subsample
mpp_data_sub = mpp_data.subsample(frac_per_obj=0.05,
add_neighborhood=True, neighborhood_size=3)

print(mpp_data.mpp.shape, mpp_data_sub.mpp.shape)


# get and plot image
import matplotlib.pyplot as plt

img = mpp_data.get_object_img(mpp_data.unique_obj_ids[0],
channel_ids=[0,1,10], img_size=255)

plt.imshow(img)
plt.show()

# add neighbors
mpp_data.add_neighborhood(size=3)
print(mpp_data.mpp.shape)



# subset to one cell
mpp_data_sub = mpp_data.subset(obj_ids=mpp_data.unique_obj_ids[:1], copy=True)

# get adata object from information in MPPData
adata = mpp_data_sub.get_adata()
print(adata)

# with adata, can use all scanpy functions, e.g. spatial plotting
import scanpy as sc
sc.pl.embedding(adata, basis='spatial', color=['00_DAPI'])