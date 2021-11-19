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

mpp_data = miann.data._data.MPPData.from_data_dir(data_dir='184A1_unperturbed/I09', data_config='NascentRNA_test')
print(mpp_data)

#subset channels
channels = [
        '01_CDK9_pT186', '01_PABPC1', '02_CDK7', '03_CDK9', '03_RPS6', '05_GTF2B', '05_Sm', '07_POLR2A', '07_SETD1A',
        '08_H3K4me3', '09_CCNT1', '09_SRRM2', '10_H3K27ac', '10_POL2RA_pS2', '11_KPNA2_MAX', '11_PML', '12_RB1_pS807_S811',
        '12_YAP1', '13_PABPN1', '13_POL2RA_pS5', '14_PCNA', '15_SON', '15_U2SNRNPB', '16_H3', '17_HDAC3', '17_SRSF2',
        '18_NONO', '19_KPNA1_MAX', '20_ALYREF', '20_SP100', '21_COIL', '21_NCL', '00_DAPI', '07_H2B'
        ]

mpp_data.subset_channels(channels)

# filter by NO_NAN cellcycle stage
# perform operation inplace
print("Cell cycle entries before subsetting by NO_NAN values:", (mpp_data.metadata.cell_cycle).unique())
mpp_data.subset(cell_cycle='NO_NAN')
print("Cell cycle entries after subsetting to NO_NAN values:", np.unique(mpp_data.metadata.cell_cycle))

#%%

# filter by specified cellcycle entries
# here, a new object is created, leaving an mpp_data object untouched
values=['G1', 'G2']
print(f"Cell cycle entries before subsetting by {values} values: {(mpp_data.metadata.cell_cycle).unique()}")
mpp_data_subset=mpp_data.subset(cell_cycle=['G1', 'G2'], copy=True)
print(f"Cell cycle entries after subsetting to {values} values: {np.unique(mpp_data_subset.metadata.cell_cycle)}")

# Subset fraction of objects:
mpp_data_sub = mpp_data.subset(frac=0.05, copy=True)
print(f"Subset fraction: {mpp_data_sub.mpp.shape[0]/mpp_data.mpp.shape[0]}")

# Subset specific number of objects:
mpp_data_sub = mpp_data.subset(num=27, copy=True)
print(f"Subset fraction: {mpp_data_sub.mpp.shape[0]/mpp_data.mpp.shape[0]}")

# subset to one cell
mpp_data_sub = mpp_data.subset(obj_ids=mpp_data.unique_obj_ids[:1], copy=True)


# add conditions
cond_params = {}
mpp_data.add_conditions(['TR_norm_lowhigh_bin_2', 'TR_bin_3', 'cell_cycle'], cond_params=cond_params)

print(mpp_data.conditions)
print(cond_params)

#%%

# filter objects with nan conditions
mpp_data.subset(nona_condition=True)

# normalise data: shift it by mean_background values(provided in channels_metadata) and then rescale each channel separately by dividing values by the corresponding 98.0 percentile.
rescale_values = []
mpp_data.normalise(background_value='mean_background', percentile= 98.0, rescale_values=rescale_values)
print(rescale_values)

# add neighbors

#first, lets subsample the data to reduce computational burden
mpp_data_sub = mpp_data.subset(frac=0.05, copy=True)
mpp_data_neighb=mpp_data_sub.add_neighborhood(size=3, copy=True)

print(f"MPP dataset before adding the neighbourhood, has neighbor data: {mpp_data_sub.has_neighbor_data}")
print(f"Representation of one MPP before adding the neighbourhood (first channel)\n: {mpp_data_sub.mpp[0,:, :, 0]}")

print(f"MPP dataset after adding the neighbourhood, has neighbor data: {mpp_data_neighb.has_neighbor_data}")
print(f"Representation of one mpp after adding the neighbourhood (first channel)\n: {mpp_data_neighb.mpp[0, :, :, 0]}")


# subsample and add neighbourhood

mpp_data_sub = mpp_data.subsample(frac_per_obj=0.05, add_neighborhood=True, neighborhood_size=3)

print(f"Subset fraction: {mpp_data_sub.mpp.shape[0]/mpp_data.mpp.shape[0]}")

print(f"MPP representation before adding the neighbourhood (first channel)\n: {mpp_data.mpp[0,:, :, 0]}")
print(f"MPP representation after adding the neighbourhood (first channel)\n: {mpp_data_sub.mpp[0, :, :, 0]}")

# Note that since a random subsampling was performed, first value in the mpp_data does not necessarily
# correspond to the first element in mpp_data_sub
