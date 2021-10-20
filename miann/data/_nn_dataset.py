from miann.constants import DATASET_DIR
from ._data import MPPData
import json
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def create_dataset(params, dataset_name, dataset_dir=DATASET_DIR):
    """
    Create a NNDataset from a params dictionary and save to dataset_name.
    """
    if params is None:
        # get params from json
        params = json.load(open(os.path.join(dataset_dir, dataset_name, 'params.json')))
    log = logging.getLogger()
    log.info('Creating train/val/test datasets with params:')
    log.info(json.dumps(params, indent=4))
    p = params
        
    mpp_datas = {'train': [], 'val': [], 'test': []}
    for data_dir in p['data_dirs']:
        mpp_data = MPPData.from_data_dir(data_dir, seed=p['seed'])
        if p['normalise']:
            mpp_data.subtract_background(p['background_value'])
        train, val, test = mpp_data.train_val_test_split(p['train_frac'], p['val_frac'])
        if p['subsample']:
            train.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'], 
                            add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
        elif p['neighborhood']:
            train.add_neighborhood(p['neighborhood_size'])
        mpp_datas['train'].append(train)
        mpp_datas['test'].append(test)
        mpp_datas['val'].append(val)
        # merge all datasets
        train = MPPData.concat(mpp_datas['train'])
        val = MPPData.concat(mpp_datas['val'])
        test = MPPData.concat(mpp_datas['test'])
        if p['normalise']:
            rescale_values = train.rescale_intensities_per_channel(percentile=p['percentile'])
            _ = val.rescale_intensities_per_channel(rescale_values=rescale_values)
            _ = test.rescale_intensities_per_channel(rescale_values=rescale_values)
            p['normalise_rescale_values'] = list(rescale_values)
        # add conditions
        if p.get('condition', None) is not None:
            train.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None),
                                 wells_metadata_file=p.get('wells_metadata_file', None))
            kwargs = {'TR_bin_3_quantile': getattr(train, 'TR_bin_3_quantile', None), 
                      'TR_lowhigh_bin_2_quantile': getattr(train, 'TR_lowhigh_bin_2_quantile', None),
                     'TR_mean_std': getattr(train, 'TR_mean_std', None)}
            val.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None),
                               wells_metadata_file=p.get('wells_metadata_file', None), **kwargs)
            test.add_conditions(p['condition'], cell_cycle_file=p.get('cell_cycle_file', None),
                                wells_metadata_file=p.get('wells_metadata_file', None), **kwargs)
            p.update(kwargs)
        # subset to valid mapobject_ids
        if p.get('subset_to_cell_cycle', False):
            train.subset(cell_cycle_file=p['cell_cycle_file'])
            val.subset(cell_cycle_file=p['cell_cycle_file'])
            test.subset(cell_cycle_file=p['cell_cycle_file'])
        # subset to non-na conditions
        if ('TR_lowhigh_bin_2' in p.get('condition', [])) or ('TR_lowhigh_bin_2' in p.get('condition', [[]])[0]):
            train.subset(condition=True)
            val.subset(condition=True)
            test.subset(condition=True)
        # subset to channels
        if p.get('channels', None) is not None:
            train.subset_channels(p['channels'])
            val.subset_channels(p['channels'])
            test.subset_channels(p['channels'])


        # get images for test and val dataset
        #test_imgs = {'img':[], 'mcu':[], 'cond': []}
        #val_imgs = {'img': [], 'mcu': [], 'cond': []}
        #test_imgs['img'] = np.array(test.get_object_imgs(data='MPP', img_size=p['test_img_size']))
        #val_imgs['img'] = np.array(val.get_object_imgs(data='MPP', img_size=p['test_img_size']))
        #if p['mcu_dir']:
        #    test_imgs['mcu'] = np.array(test.get_object_imgs(data='MCU', img_size=p['test_img_size']))
        #    val_imgs['mcu'] = np.array(val.get_object_imgs(data='MCU', img_size=p['test_img_size']))
        #if p.get('condition', None) is not None:
        #    test_imgs['cond'] = np.array(test.get_object_imgs(data='condition', img_size=p['test_img_size']))
        #    val_imgs['cond'] = np.array(val.get_object_imgs(data='condition', img_size=p['test_img_size']))
        test_imgs = test.copy()
        val_imgs = val.copy()
        # subsample and add neighbors to val and test
        if p['subsample']:
            val.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'], 
                                add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
            test.subsample(frac=p['frac'], frac_per_obj=p['frac_per_obj'], num=p['num'], num_per_obj=p['num_per_obj'], 
                                add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'])
        elif p['neighborhood']:
            test.add_neighborhood(p['neighborhood_size'])
            val.add_neighborhood(p['neighborhood_size'])
            
        log.info('-------------------')
        log.info('created datasets:')
        log.info('train: {}'.format(str(train)))
        log.info('val: {}'.format(str(val)))
        log.info('test: {}'.format(str(test)))
        log.info('-------------------')

        # save data
        outdir = os.path.join(dataset_dir, dataset_name)
        os.makedirs(outdir, exist_ok=True)
        train.write(os.path.join(outdir, 'train'))
        val.write(os.path.join(outdir, 'val'))
        test.write(os.path.join(outdir, 'test'))
        val_imgs.write(os.path.join(outdir, 'val_imgs'))
        test_imgs.write(os.path.join(outdir, 'test_imgs'))

        # save params
        json.dump(params, open(os.path.join(outdir, 'params.json'), 'w'), indent=4)


class NNDataset:
    """
    Dataset for training and evaluation of neural networks.

    A NNDataset is stored train/val/test/val_img/test_img folders that contain MPPData.
    """
    def __init__(self, dataset_name, dataset_dir=DATASET_DIR):
        self.log = logging.getLogger(self.__class__.__name__)
        self.dataset_folder = os.path.join(dataset_dir, dataset_name)
        
        # data
        self.data = {
            'train': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'train')),
            'val': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'val')),
            'test': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'test'))
        }
        self.imgs = {
            'val': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'val_imgs')),
            'test': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'test_imgs'))
        }
        self.channels = self.data['train'].channels.reset_index().set_index('name')
        
    def get_channel_ids(self, to_channels, from_channels=None):
        if from_channels is None:
            from_channels = self.channels.copy()
            from_channels['index'] = np.arange(len(from_channels))
        if not isinstance(from_channels, pd.DataFrame):
            from_channels = pd.DataFrame({'name': from_channels, 'index': np.arange(len(from_channels))})
            from_channels = from_channels.set_index('name')
        from_channels = from_channels.reindex(to_channels)
        return list(from_channels['index'])

    def x(self, split, is_conditional=False):
        x = self.data[split].mpp.astype(np.float32)
        if is_conditional:
            c = self.data[split].conditions.astype(np.float32)
            x = (x, c)
        return x

    def y(self, split, output_channels=None):
        y = self.data[split].center_mpp
        if output_channels is not None:
            channel_ids = self.get_channel_ids(output_channels)
            y = y[:,channel_ids]
        return y

    def get_tf_dataset(self, split='train', output_channels=None, is_conditional=False, 
                       repeat_y=False, add_c_to_y=False, shuffled=False):
        """returns tf.data.Dataset of the desired split.
        
        shuffled: for generator dataset, shuffle indices before generating data.
        will produce same order every time 

        repeat_y: match output len to number of losses 
        (otherwise keras will not work, even if its losses that do not need y)

        add_c_to_y: append condition to y. Needed for adversarial loss
        """

        output_types = [tf.float32, tf.float32]
        output_shapes = []

        # x
        x = self.x(split, is_conditional)
        if is_conditional:
            num = x[0].shape[0]
            output_types.append(tuple([tf.float32, tf.float32]))
            output_shapes.append(tuple(tf.TensorShape(x[0].shape[1:]), tf.TensorShape(x[1].shape[1:])))
        else:
            num = x.shape[0]
            output_types.append(tf.float32)
            output_shapes.append(tf.TensorShape(x.shape[1:]))
        
        # y
        y = self.y(split, output_channels)
        output_types.append(tf.float32)
        output_shapes.append(tf.TensorShape(y.shape[1:]))
        if repeat_y is not False:  # TODO concat c here instead of y! (for adv loss)
            output_types[1] = tuple([tf.float32 for _ in range(repeat_y)])
            y = tuple([y for _ in range(repeat_y)])
            output_shapes[1] = tuple([output_shapes[1] for _ in range(repeat_y)])
        if add_c_to_y is not False:
            assert is_conditional
            # get output_type and shape for c from first output
            c_output_type = output_types[0][1]
            c_output_shape = output_shapes[0][1]
            # add c to y and output data to types and shapes
            if isinstance(output_types[1], tuple):
                output_types[1] = tuple(list(output_types[1]) + [c_output_type])
                y = tuple(list(y) + [x[1]])
                output_shapes[1] = tuple(list(output_shapes[1]) + [c_output_shape])
            else:
                output_types[1] = tuple([output_types[1], c_output_type])
                y = tuple([y, x[1]])
                output_shapes[1] = tuple([output_shapes[1], c_output_shape])
        
        # create a generator dataset:
        indices = np.arange(num)
        if shuffled:
            rng = np.random.default_rng(seed=0)
            rng.shuffle(indices)
        def gen():
            for i in indices:
                if is_conditional:
                    el_x = (x[0][i], x[1][i])
                else:
                    el_x = x[i]
                if repeat_y is not False:
                    el_y = tuple([y[j][i] for j in range(len(y))])
                else:
                    el_y = y[i]
                yield (el_x, el_y)
        dataset = tf.data.Dataset.from_generator(gen, tuple(output_types), tuple(output_shapes))
        return dataset
    
    # TODO: function to create inputs for imgs? Maybe very easy, and do not need here?
    # input_data = (val_imgs.mpp, val_imgs.conditions)

    #def get_mapobject_ids(self, split='train', data='data'):
    #    if data == 'data':
    #        return self.data[split]['mapobject_id']
    #    else:
    #        return self.imgs[split]['mapobject_id']
    
    #def get_imgs(self, split='val', img_ids=None, is_conditional=False):
    #    if img_ids is None:
    #        img_ids = np.arange(len(self.imgs[split]['img']))
    #    # randomly select img_ids
    #    if not isinstance(img_ids, np.ndarray):
    #        np.random.seed(42)
    #        img_ids = np.random.choice(range(len(self.imgs[split]['img'])), img_ids)
    #    imgs = self.imgs[split]['img'][img_ids]
    #    cond = None
    #    if is_conditional:
    #        cond = self.imgs[split]['cond'][img_ids]
    #    return (imgs, cond), img_ids
    
    #def get_metadata(self, split, columns=['mapobject_id', 'well_name', 'cell_type', 'perturbation_duration', 'cell_cycle']):
    #    mapobject_ids = self.get_mapobject_ids(split)
    #    wells = pd.read_csv(os.path.join(DATA_DIR, 'wells_metadata.csv'), index_col=0)
    #    cc = pd.read_csv(os.path.join(DATA_DIR, 'cell_cycle_classification.csv'))
    #    metadata = self.metadata.set_index('mapobject_id').loc[mapobject_ids]
    #    metadata = metadata.reset_index()
    #    metadata = metadata.merge(wells, left_on='well_name', right_on='well_name', how='left', suffixes=('','well_'))
    #    metadata = metadata.merge(cc, left_on='mapobject_id_cell', right_on='mapobject_id', how='left', suffixes=('','cc_'))
    #    return metadata[columns]