

# prepare data for loading with MPPData
# need one combined metadata file
from miann_ana import merge_mpp_metadata
merge_mpp_metadata(metadata, cell_cycle, wells_metadata)

# create dataset (uses MPPData)
from miann.data import create_dataset, MPPDataset
create_dataset(params, folder)  # saves MPPdata in subfolders
# use dataset for training:
MPPDataset(folder).get_tf_dataset(split)
# also contains val_imgs and test_imgs as MPPdata


