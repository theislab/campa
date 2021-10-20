from abc import ABC, abstractmethod
import numpy as np
from logging import getLogger
from miann.constants import OBJ_ID
import pandas as pd
import os
from miann.data._img_utils import pad_border, BoundingBox

class BaseData(ABC):

    def __init__(self, metadata, channels, **kwargs):
        self.metadata = metadata
        self.channels = channels
        self._is_subsampled = False
        self.seed = kwargs.get('seed', 42)
        self.rng = np.random.default_rng(seed=self.seed)
        self.log = getLogger(self.__class__.__name__)
        return super().__init__()

    @classmethod
    @abstractmethod
    def from_data_dir(cls, data_dir, **kwargs):
        pass

    @property
    def is_subsampled(self):
        """
        True, if this Data instance has been subsampled and does not contain all observations (pixels)
        for every present object anymore
        """
        return self._is_subsampled

    @abstractmethod
    def get_obj_img(self, obj_id, **kwargs):
        """
        Extract single cell (object) image using unique id.
        Only works for BaseData instance that have not been subsampled
        """
        pass

    def subset_objects(self, key, value):
        """
        Restrict objects to those with specified value(s) for key in the metadata table

        Args:
            key: name of column used to select objects
            value (str or list of str): allowed entries for selected objects

        Returns:
            nothing, modifies Data object in place
        """
        # TODO implement
        pass

    def subset_channels(self, channels):
        # TODO implement
        pass


class ImageData(BaseData):

    def __init__(self, metadata, channels, imgs={}, seg_imgs={}, **kwargs):
        super().__init__(metadata, channels, **kwargs)
        self.imgs = imgs
        self.seg_imgs = seg_imgs

    @classmethod
    def from_dir(cls, data_dir):
        # TODO read metadata and images (lazy) and call constructuor
        pass

    def get_obj_img(self, obj_id, kind='data'):
        # TODO crop and return object images for different kinds: data, mask (object mask)
        pass


class MPPData(BaseData):

    #required_keys = ['x' , 'y', 'mpp', 'obj_ids']
    #optional_keys = ['conditions', 'labels', 'icl', 'latent']

    def __init__(self, metadata, channels, data, **kwargs):
        super().__init__(metadata, channels, **kwargs)
        for required_key in ['x' , 'y', 'mpp', 'obj_ids']:
            assert required_key in data.keys(), f"required key {required_key} missing from data"
        self._data = data
        # subset metadata to obj_ids
        self.metadata = metadata[metadata[OBJ_ID].isin(np.unique(self.obj_ids))]
        if len(self.mpp.shape) == 2:
            self._data['mpp'] = self.mpp[:,np.newaxis,np.newaxis,:]
        self.log.info(f'created new MPPData with {len(self.metadata)} objects')

    @classmethod
    def from_image_data(self, image_data: ImageData):
        # TODO convert ImageData to MPPData 
        pass

    @classmethod
    def from_data_dir(cls, data_dir, mode='r', **kwargs):
        """
        Read MPPData from directory.

        Expects the following files: x.npy, y.npy, mpp.npy, mapobject_ids.npy
        If present, will read the following additional files: 
            labels.npy, icl.npy, latent.npy

        Args:
            mode: mmap_mode for np.load. Set to None to load data in memory.
        """
        # read all data from data_dir
        metadata = pd.read_csv(os.path.join(data_dir, 'metadata.csv'), index_col=0).reset_index(drop=True)
        channels = pd.read_csv(os.path.join(data_dir, 'channels.csv'), names=['index', 'name'], index_col=0).reset_index(drop=True)
        # read npy data
        data = {}
        for fname in ['x', 'y', 'mpp']:
            data[fname] = np.load(os.path.join(data_dir, f'{fname}.npy'), mmap_mode='r')
        try:  # TODO this is legacy code, if creating new data, do not need
            data['obj_ids'] = np.load(os.path.join(data_dir, 'mapobject_ids.npy'))
        except FileNotFoundError as e:
            data['obj_ids'] = np.load(os.path.join(data_dir, 'obj_ids.npy'))
        for fname in ['labels', 'icl', 'latent']:
            try:
                data[fname] = np.load(os.path.join(data_dir, f'{fname}.npy'))
            except FileNotFoundError as e:
                continue
        # init self
        self = cls(metadata=metadata, channels=channels, data=data, **kwargs)
        self.data_dir = data_dir
        return self
    
    @classmethod
    def concat(cls, objs):
        """concatenate the mpp_data objects by concatenating all arrays and return a new one"""
        # channels and _data.keys() need to be the same
        for mpp_data in objs:
            assert (mpp_data.channels.name == objs[0].channels.name).all()
            assert (mpp_data._data.keys() == objs[0]._data.keys()).all()

        channels = objs[0].channels
        # concatenate metadata (pandas)
        metadata = pd.concat([mpp_data.metadata for mpp_data in objs], axis=0, ignore_index=True)
        # concatenate numpy arrays
        data = {}
        for key in objs[0]._data.keys():
            data[key] = np.concatenate([mpp_data._data[key] for mpp_data in objs], axis=0)
        
        self = cls(metadata=metadata, channels=channels, data=data, seed=objs[0].seed)
        self.log.info('Concatenated several MPPDatas')
        return self

    # --- Properties ---
    @property
    def mpp(self):
        return self._data['mpp']

    @property
    def obj_ids(self):
        return self._data['obj_ids']

    @property
    def latent(self):
        if 'latent' in self._data.keys():
            return self._data['latent']
        return None

    @property
    def icl(self):
        if 'icl' in self._data.keys():
            return self._data['icl']
        return None

    @property
    def conditions(self):
        if 'conditions' in self._data.keys():
            return self._data['conditions']
        return None

    @property
    def has_neighbor_data(self):
        return (self.mpp.shape[1]!=1) and (self.mpp.shape[2]!=1)
    
    @property
    def center_mpp(self):
        c = self.mpp.shape[1]//2
        return self.mpp[:,c,c,:]

    @property
    def unique_obj_ids(self):
        return np.unique(self.obj_ids)

    def __str__(self):
        return 'MPPData ({} mpps with shape {} from {} objects)'.format(self.mpp.shape[0], self.mpp.shape[1:], len(self.metadata))
    
    # --- Modify / Add data ---
    def apply_mask(self, mask, copy=False):
        """
        return new MPPData with masked self._data values
        """
        # TODO use this fn for all sorts of masking
        data = {}
        for key in self._data.keys():
            data[key] = self._data[key][mask]
        
        if copy is False:
            self._data = data
            self.metadata = self.metadata[self.metadata[OBJ_ID].isin(np.unique(self.obj_ids))]
        else:
            return MPPData(metadata=self.metadata, channels=self.channels, data=data, seed=self.seed)

    def add_exp_data(self, exp):
        # TODO add self._data['latent'] and self._data['clustering'] from files saved in "aggregated" experiment folder
        # TODO check that ids are correct, and lengths are the same
        pass
    
    def train_val_test_split(self, train_frac=0.8, val_frac=0.1):
        """split along mapobject_ids for train/val/test split"""
        # TODO (maybe) adapt and ensure that have even val/test fractions from each well
        ids = self.unique_obj_ids.copy()
        self.rng.shuffle(ids)
        num_train = int(len(ids)*train_frac)
        num_val = int(len(ids)*val_frac)
        train_ids = ids[:num_train]
        val_ids = ids[num_train:num_train+num_val]
        test_ids = ids[num_train+num_val:]
        self.log.info('splitting data in {} train, {} val, and {} test objects'.format(len(train_ids), len(val_ids), len(test_ids)))
        splits = []
        for split_ids in (train_ids, val_ids, test_ids):
            ind = np.in1d(self.mapobject_ids, split_ids)
            splits.append(self.apply_mask(ind, copy=True))
        return splits

    # --- Saving ---
    def write(self, save_dir):
        """
        Write MPPData to disk.
        
        Save channels, metadata as csv and one npy file per entry in self._data.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log.info(f'Saving mpp data to {save_dir}')
        for key in self._data.keys():
            np.save(os.path.join(save_dir, f'{key}.npy'), self._data[key])
        self.channels.to_csv(os.path.join(save_dir, 'channels.csv'), header=None, index=None)
        self.metadata.to_csv(os.path.join(save_dir, 'metadata.csv'))
    
    
    def copy(self):
        #TODO
        pass

    # TODO: functions for easy access to conditions
    # TODO: check / adapt below old functions


    #---- Functions for modify the MPPData in place ----
            
    def _get_per_mpp_value(self, per_cell_value):
        """takes list of values corresponding to self.metadata.mapobject_id
        and propagates them to self.mapobject_id"""
        per_cell_df = pd.DataFrame({'val': per_cell_value, 'mapobject_id': self.metadata.mapobject_id})
        df = pd.DataFrame({'mapobject_id': self.mapobject_ids})
        per_mpp_value = df.merge(per_cell_df, left_on='mapobject_id', right_on='mapobject_id', how='left')['val']
        return per_mpp_value
    
    def get_condition(self, desc, cell_cycle_file=None, wells_metadata_file=None, **kwargs): # TODO add well_metadata_file for adding conditions to simulated data
        """
        return condition based on cond (used by add_conditions).
        If cond is a list of conditions, return unique one-hot encoded vector combining multiple conditions 
        (only possible when all sub-conditions are one-hot encoded)
        """
        cond = None
        if isinstance(desc, list):
            conds = [self.get_condition(d, cell_cycle_file=cell_cycle_file, **kwargs) for d in desc]
            # check if combining is possible
            #assert np.all([np.all(np.unique(c) == [0,1]) for c in conds]), f"one of cond is not one-hot: {desc}"
            cond = get_combined_one_hot(conds)
        elif np.any([el in desc for el in ['cell_cycle', 'perturbation', 'well_name', 'siRNA']]):
            self.log.info(f'getting condition {desc}')
            # read cell cycle and well file and match with mapobjectids
            cell_cycle = pd.read_csv(cell_cycle_file)
            if wells_metadata_file is None:
                wells_metadata_file = os.path.join(DATA_DIR, 'wells_metadata.csv')
            wells = pd.read_csv(wells_metadata_file)
            df = self.metadata.merge(cell_cycle, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','_cc'))
            df = df.merge(wells, left_on='well_name', right_on='name', how='left', suffixes=('', '_well'))
            # get correct desc name (match to column names)
            one_hot = 'one_hot' in desc
            desc = desc.replace('_one_hot', '') if one_hot else desc
            if 'perturbation' in desc:
                desc = 'perturbation' if desc == 'perturbation2' else 'perturbation_duration'
                # NOTE maps perturbation to perturbation_duration and perturbation2 to perturbation
            self.log.info(f'looking up condition {desc}')
            cond = np.array(df[desc])
            cond = self._get_per_mpp_value(cond)
            if one_hot:
                cond = convert_condition(np.array(cond), desc=desc, one_hot=True)
            else:
                cond = convert_condition(np.array(cond)[:,np.newaxis], desc=desc)
        elif desc in list(self.channels.name) or 'TR' in desc:
            if desc in list(self.channels.name):
                # desc is channel - get values for this channel
                cid = list(self.channels.name).index(desc)
                # calculate mean channel value per cell
                df = pd.DataFrame({'cond': self.center_mpp[:,cid], 'mapobject_ids': self.mapobject_ids})
                df_mean = df.groupby('mapobject_ids').mean() #TODO could use _get_per_mpp_value
                cond = np.array(df.merge(df_mean, right_index=True, left_on='mapobject_ids', how='left')['cond_y'])
            else:
                if desc not in ['TR', 'TR_norm', 'TR_zscore', 'TR_norm_zscore', 'TR_bin_3', 'TR_lowhigh_bin_2']:
                    raise NotImplementedError(desc)
                
                # desc is TR / TR_norm (in metadata)
                # propagate per-mapobject_id values to pixels
                short_desc = 'TR_norm' if 'TR_norm' in desc else 'TR'
                if short_desc not in self.metadata.columns:
                    short_desc = short_desc + '_x'
                self.log.info(f'using column {short_desc} in metadata')
                cond = self._get_per_mpp_value(self.metadata[short_desc])
                
            # post process TR
            if 'bin' in desc:
                if desc == 'TR_bin_3':
                    # bin in .33 and .66 quantiles (3 classes)
                    if kwargs.get('TR_bin_3_quantile', None) is not None:
                        q = kwargs['TR_bin_3_quantile']
                    else:
                        q = np.quantile(cond, q=(0.33, 0.66))
                        self.TR_bin_3_quantile = list(q)
                    cond_bin = np.zeros_like(cond).astype(int)
                    cond_bin[cond > q[0]] = 1
                    cond_bin[cond > q[1]] = 2
                    cond = get_one_hot(cond_bin, nb_classes=3)

                elif desc == 'TR_lowhigh_bin_2':
                    # bin in 4 quantiles, take low and high TR cells (2 classes)
                    # remainder of cells has nan values - can be filtered out later
                    if kwargs.get('TR_lowhigh_bin_2_quantile', None) is not None:
                        q = kwargs['TR_lowhigh_bin_2_quantile']
                    else:
                        q = np.quantile(cond, q=(0.25, 0.75))
                        self.TR_lowhigh_bin_2_quantile = list(q)
                    cond_bin = np.zeros_like(cond).astype(int)
                    cond_bin[cond > q[1]] = 1
                    cond_one_hot = get_one_hot(cond_bin, nb_classes=2)
                    cond_one_hot[np.logical_and(cond > q[0], cond < q[1])] = np.nan
                    cond = cond_one_hot
            else:
                if 'zscore' in desc:
                    # z-score TR
                    if kwargs.get('TR_mean_std', None) is not None:
                        tr_mean, tr_std = kwargs['TR_mean_std']
                    else:
                        tr_mean, tr_std = cond.mean(), cond.std()
                        self.TR_mean_std = [tr_mean, tr_std]
                    cond = (cond - tr_mean) / tr_std
                    
                # add empty axis for concatenation
                cond = cond[:, np.newaxis]
        else:
            NotImplementedError(desc)
        return cond
            
    def add_conditions(self, cond_desc, cell_cycle_file=None, **kwargs):
        """
        Add conditions informations by aggregating over channels (per cell) or reading data from cell cycle file.
        Implemented conditions: '00_EU' and 'cell_cycle'
        """
        self.log.info(f'Adding conditions: {cond_desc}')
        conditions = []
        for desc in cond_desc:
            cond = self.get_condition(desc, cell_cycle_file=cell_cycle_file, **kwargs)
            conditions.append(cond)
        self.conditions = np.concatenate(conditions, axis=-1)

            
    def subset_objects(self, cell_cycle_file=None, rand_frac=None, condition=None):
        """
        Restrict objects to those with specified value(s) for key in the metadata table

        Args:
            key: name of column used to select objects
            value (str or list of str): allowed entries for selected objects

        Returns:
            nothing, modifies Data object in place
        """

        """\
        Subset objects to mapobject_ids listed in fname. Use 
        """
        # TODO also allow subsetting to list of mapobject ids
        mask = np.ones(len(self.mpp), dtype=np.bool)
        if cell_cycle_file is not None:
            cell_cycle = pd.read_csv(cell_cycle_file)
            cond = np.array(self.metadata.merge(cell_cycle, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','_cc'))['cell_cycle'])
            cond = self._get_per_mpp_value(cond)
            mask &= ~cond.isnull()
            self.log.info(f'Subsetting mapobject ids to cell_cycle_file (removing {sum(cond.isnull())} objects)')
        elif rand_frac is not None:
            num_mapobject_ids = int(len(self.metadata)*rand_frac)
            selected_mapobject_ids = np.zeros(len(self.metadata), dtype=bool)
            np.random.seed(42)
            selected_mapobject_ids[np.random.choice(len(self.metadata), size=num_mapobject_ids, replace=False)] = True
            mask = self._get_per_mpp_value(selected_mapobject_ids)
            self.log.info(f'Subsetting mapobject ids from {len(self.metadata)} to {num_mapobject_ids} cells')
        elif condition:
            # subset to non-nan conditions
            mask = ~np.isnan(self.conditions).any(axis=1)
            self.log.info(f'Removing nan conditions - Subsetting mapobject ids from {len(self.conditions)} to {sum(mask)} mpps')
        else:
            self.log.warn('Called subset, not specified what to subset to')
            
        # apply mask to data in self
        self.labels = self.labels[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.mpp = self.mpp[mask]
        self.mcu_ids = self.mcu_ids[mask]
        self.mapobject_ids = self.mapobject_ids[mask]
        self.metadata = self.metadata[self.metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]
        self.conditions = self.conditions[mask]
        
        return mask
        
            
    def subset_channels(self, channels):
        """
        Restrict self.mpp to defined channels. Channels are given as string values.
        Updates self.mpp and self.channels
        """
        cids = list(self.channels.set_index('name').loc[channels]['channel_id'])
        self.channels = self.channels.loc[cids] # TODO reset index
        self.mpp = self.mpp[:,:,:,cids]
        self.log.info('restricted channels to {} channels'.format(len(self.channels)))
        
    def subsample(self, frac=None, frac_per_obj=None, num=None, num_per_obj=None, add_neighborhood=False, neighborhood_size=3):
        """
        Subsample MPPData based on selecting mpps. 
        All other information is updated accordingly (to save RAM/HDD-memory)
        """
        assert sum([frac!=None, frac_per_obj!=None, num!=None, num_per_obj!=None]) == 1, "set only one of the params to a value"
        assert not (self.has_neighbor_data and add_neighborhood), "cannot add neighborhood, already has neighbor data"
        if frac is not None:
            num = int(len(self.mpp)*frac)
        if num is not None:
            self.log.info('subsampling data to {} (from {})'.format(num, len(self.mpp)))
            # randomly select num mpps
            # NOTE: need to seed default_rng, np.random.seed does not work 
            rng = np.random.default_rng(seed=self.seed)
            selected = rng.choice(len(self.mpp), num, replace=False)
            # select other information accordingly
            labels = self.labels[selected]
            x = self.x[selected]
            y = self.y[selected]
            mpp = self.mpp[selected]
            mapobject_ids = self.mapobject_ids[selected]
            mcu_ids = self.mcu_ids[selected]
            conditions = self.conditions[selected]
        if frac_per_obj is not None or num_per_obj is not None:
            self.log.info('subsampling data with frac_per_obj/num_per_obj')
            # iterate over all object_ids
            labels = []
            x = []
            y = []
            mpp = []
            mapobject_ids = []
            mcu_ids = []
            conditions = []
            for mapobject_id in np.unique(self.mapobject_ids):
                obj_mask = self.mapobject_ids == mapobject_id
                if frac_per_obj is not None:
                    cur_num = int(obj_mask.sum() * frac_per_obj)
                else:
                    cur_num = num_per_obj
                rng = np.random.default_rng(seed=self.seed)
                selected = rng.choice(obj_mask.sum(), cur_num, replace=False)
                # select
                labels.append(self.labels[obj_mask][selected])
                x.append(self.x[obj_mask][selected])
                y.append(self.y[obj_mask][selected])
                mpp.append(self.mpp[obj_mask][selected])
                mapobject_ids.append(self.mapobject_ids[obj_mask][selected])
                mcu_ids.append(self.mcu_ids[obj_mask][selected])
                conditions.append(self.conditions[obj_mask][selected])
            labels = np.concatenate(labels, axis=0)
            x = np.concatenate(x, axis=0)
            y = np.concatenate(y, axis=0)
            mpp = np.concatenate(mpp, axis=0)
            mapobject_ids = np.concatenate(mapobject_ids, axis=0)
            mcu_ids = np.concatenate(mcu_ids, axis=0)
            conditions = np.concatenate(conditions, axis=0)
        if add_neighborhood:
            self.log.info('adding neighborhood of size {}'.format(neighborhood_size))
            neighbor_mpp = self.get_neighborhood(mapobject_ids, x, y, size=neighborhood_size)
            #assert (neighbor_mpp[:,neighborhood_size//2,neighborhood_size//2,:] == mpp[:,0,0,:]).all()
            mpp = neighbor_mpp
        # update self
        self.labels=labels
        self.x=x
        self.y=y
        self.mpp=mpp
        self.mapobject_ids=mapobject_ids
        self.mcu_ids=mcu_ids
        self.conditions=conditions
        self.metadata = self.metadata[self.metadata.mapobject_id.isin(np.unique(self.mapobject_ids))]
    
    def add_neighborhood(self, size=3):
        assert not self.has_neighbor_data, "cannot add neighborhood, already has neighbor data"
        self.log.info('adding neighborhood of size {}'.format(size))
        mpp = self.get_neighborhood(self.mapobject_ids, self.x, self.y, size=size)
        assert (mpp[:,size//2,size//2,:] == self.center_mpp).all()
        self.mpp = mpp
        
    def subtract_background(self,background_value):
        # code copied/adapted from Scott Berry's MCU package
        # Note mpp is converted to float
        if isinstance(background_value, float):
            self.log.info('subtracting constant value of {} from all channels'.format(background_value))
            self.mpp == self.mpp.astype(np.float32) - background_value
        else:
            self.log.debug('reading channel-specific background from {}'.format(background_value))
            bkgrd = pd.read_csv(background_value).merge(self.channels, left_on='channel', right_on='name', how='right')
            bkgrd = bkgrd.loc[bkgrd['measurement_type'] == "non-cell"].loc[bkgrd['cell_line']== "HeLa"]
            bkgrd = bkgrd[['channel_id','mean_background']]

            # create a dictionary to link mpp columns with their background values
            bkgrd_dict = bkgrd.set_index('channel_id')['mean_background'].to_dict()
            # check all channels are present
            if len(bkgrd_dict) != self.channels.shape[0]:
                missing_channels = list(set(self.channels.channel_id).difference(bkgrd_dict.keys()))
                self.log.warning('missing background value for channels {}'.format(list(self.channels.loc[missing_channels].name)))
                
            # subtract per-channel background (row-wise) from mpp
            # if background not knows, subtract 0
            bkgrd_vec = np.array(
                [bkgrd_dict.get(i, 0) for i in range(0,self.channels.shape[0])]).astype(np.float64)

            self.log.info('subtracting channel-specific background: {}'.format(
                ', '.join([str(el) for el in bkgrd_vec.tolist()])
            ))
            self.mpp = self.mpp.astype(np.float64) - bkgrd_vec
        # cut off at 0 (no negative values)
        self.mpp[self.mpp<0] = 0
        
    def rescale_intensities_per_channel(self,percentile=98.0,rescale_values=None):
        # Note mpp is modified in place and function returns None
        if rescale_values is None:
            rescale_values = np.percentile(self.center_mpp, percentile, axis=0)
        self.log.info('rescaling mpp intensities per channel with values {}'.format(rescale_values))
        self.mpp = self.mpp / rescale_values
        self.log.info('converting mpp values to float32')
        self.mpp = self.mpp.astype(np.float32)
        return rescale_values

    # ---- getting functions -----
    
    def get_neighborhood(self, mapobject_ids, xs, ys, size=3, border_mode='center'):
        """return neighborhood information for given mapobject_ids + xs + ys"""
        data = np.zeros((len(mapobject_ids), size, size, len(self.channels)), dtype=self.mpp.dtype)
        for mapobject_id in np.unique(mapobject_ids):
            mask = mapobject_ids == mapobject_id
            img, (xoff, yoff) = self.get_mpp_img(mapobject_id, pad=size//2)
            vals = []
            for x, y in zip(xs[mask], ys[mask]):
                idx = tuple(slice(pp-size//2, pp+size//2+1) for pp in [y-yoff,x-xoff])
                vals.append(pad_border(img[idx], mode=border_mode))
            data[mask] = np.array(vals)
        return data
    
    def get_img_from_data(self, x, y, data, img_size=None, pad=0):
        """
        Create image from x and y coordinates and fill with data.
        Args:
            x, y: 1-d array of x and y corrdinates
            data: shape (n_coords,n_channels), data for the image
            img_size: size of returned image
            pad: amount of padding added to returned image (only used when img_size is None)
        """
        x_coord = x - x.min() + pad
        y_coord = y - y.min() + pad
        # create image
        img = np.zeros((y_coord.max()+1+pad, x_coord.max()+1+pad, data.shape[-1]), dtype=data.dtype)
        img[y_coord,x_coord] = data
        # resize
        if img_size is not None:
            c = BoundingBox(0,0,img.shape[0], img.shape[1]).center
            bbox = BoundingBox.from_center(c[0], c[1], img_size, img_size)
            img = bbox.crop(img)
            return img
        else:
            # padding info is only relevant when not cropping img to shape
            return img, (x.min()-pad, y.min()-pad)
    
    def get_mcu_img(self, mapobject_id, **kwargs):
        """
        Calculate MCU image of given mapobject
        kwargs: arguments for get_img_from_data
        """
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.mcu_ids[mask][:,np.newaxis]
        return self.get_img_from_data(x, y, data, **kwargs)
    
    def get_mpp_img(self, mapobject_id, channel_ids=None, **kwargs):
        """
        Calculate MPP image of given mapobject
        channel_ids: ids of MPP channels that the image should hav. If None, all channels are returned.
        kwargs: arguments for get_img_from_data
        """     
        if channel_ids is None:
            channel_ids = range(len(self.channels))
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.center_mpp[mask][:,channel_ids]
        return self.get_img_from_data(x, y, data, **kwargs)

    def get_obj_img(self, **kwargs):
        pass
    
    def get_condition_img(self, mapobject_id, **kwargs):
        """
        Calculate condition image of given mapobject
        kwargs: arguments for get_img_from_data
        """
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = self.conditions[mask]
        return self.get_img_from_data(x, y, data, **kwargs)
    
    def get_cluster_img(self, mapobject_id, cluster_ids, **kwargs):
        mask = self.mapobject_ids == mapobject_id
        x = self.x[mask]
        y = self.y[mask]
        data = cluster_ids[mask][:,np.newaxis]
        return self.get_img_from_data(x, y, data, **kwargs)
       
    def get_object_imgs(self, data='MPP', channel_ids=None, img_size=None, pad=0):
        imgs = []
        for mapobject_id in self.metadata.mapobject_id:
            if data == 'MPP':
                res = self.get_mpp_img(mapobject_id, channel_ids, img_size=img_size, pad=pad)
            elif data == 'MCU':
                res = self.get_mcu_img(mapobject_id, img_size=img_size, pad=pad)
            elif data == 'condition':
                res = self.get_condition_img(mapobject_id, img_size=img_size, pad=pad)
            else:
                raise NotImplementedError
            if img_size is None:
                res = res[0]
            imgs.append(res)
        return imgs

