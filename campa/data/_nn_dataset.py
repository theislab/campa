from campa.constants import get_data_config
from ._data import MPPData
import json
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def create_dataset(params):
    """
    Create a NNDataset from params and save to dataset_name (defined in params).
    """
    log = logging.getLogger()
    log.info('Creating train/val/test datasets with params:')
    log.info(json.dumps(params, indent=4))
    p = params
    # prepare outdir
    data_config = get_data_config(p['data_config'])
    outdir = os.path.join(data_config.DATASET_DIR, p['dataset_name'])
    os.makedirs(outdir, exist_ok=True)
    # prepare datasets
    mpp_datas = {'train': [], 'val': [], 'test': []}
    for data_dir in p['data_dirs']:
        mpp_data = MPPData.from_data_dir(data_dir, seed=p['seed'], data_config=p['data_config'])
        train, val, test = mpp_data.train_val_test_split(**p['split_kwargs'])
        # subsample train data now
        if p['subsample']:
            train = train.subsample(add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'], 
                **p['subsample_kwargs'])
        elif p['neighborhood']:
            train.add_neighborhood(p['neighborhood_size'])
        mpp_datas['train'].append(train)
        mpp_datas['test'].append(test)
        mpp_datas['val'].append(val)
    # merge all datasets
    train = MPPData.concat(mpp_datas['train'])
    val = MPPData.concat(mpp_datas['val'])
    test = MPPData.concat(mpp_datas['test'])
    # prepare (channels, normalise, condition, subset)
    train.prepare(params)  # this has side-effects on params, st val + test use correct params
    val.prepare(params)
    test.prepare(params)
    # save test and val imgs
    val.write(os.path.join(outdir, 'val_imgs'))
    test.write(os.path.join(outdir, 'test_imgs'))
    # subsample and add neighbors to val and test (for prediction during training)
    if p['subsample']:
        val = val.subsample(add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'], 
                **p['subsample_kwargs'])
        test = test.subsample(add_neighborhood=p['neighborhood'], neighborhood_size=p['neighborhood_size'], 
                **p['subsample_kwargs'])
    elif p['neighborhood']:
        val.add_neighborhood(p['neighborhood_size'])
        test.add_neighborhood(p['neighborhood_size'])
            
    log.info('-------------------')
    log.info('created datasets:')
    log.info('train: {}'.format(str(train)))
    log.info('val: {}'.format(str(val)))
    log.info('test: {}'.format(str(test)))
    log.info('-------------------')

    # save datasets
    train.write(os.path.join(outdir, 'train'))
    val.write(os.path.join(outdir, 'val'))
    test.write(os.path.join(outdir, 'test'))
    # save params
    json.dump(params, open(os.path.join(outdir, 'params.json'), 'w'), indent=4)


class NNDataset:
    """
    Dataset for training and evaluation of neural networks.

    A NNDataset is stored train/val/test/val_img/test_img folders that contain MPPData.
    """
    def __init__(self, dataset_name, data_config=None):
        self.log = logging.getLogger(self.__class__.__name__)
        if data_config is None:
            self.data_config_name = "NascentRNA"
            self.log.warn(f"Using default data_config {self.data_config_name}")
        else:
            self.data_config_name = data_config
        self.data_config = get_data_config(self.data_config_name)
        self.dataset_folder = os.path.join(self.data_config.DATASET_DIR, dataset_name)
        
        # data
        self.data = {
            'train': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'train'), base_dir=''),
            'val': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'val'), base_dir=''),
            'test': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'test'), base_dir='')
        }
        self.imgs = {
            'val': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'val_imgs'), base_dir=''),
            'test': MPPData.from_data_dir(os.path.join(self.dataset_folder, 'test_imgs'), base_dir='')
        }
        self.channels = self.data['train'].channels.reset_index().set_index('name')
        self.params = json.load(open(os.path.join(self.dataset_folder, 'params.json'), 'r'))

    def __str__(self):
        s = f"NNDataset for {self.data_config_name} (shape {self.data['train'].mpp.shape[1:]})."
        s += f" train: {len(self.data['train'].mpp)}, val: {len(self.data['val'].mpp)}, test: {len(self.data['test'].mpp)}"
        return s

    def x(self, split, is_conditional=False):
        x = self.data[split].mpp.astype(np.float32)
        if is_conditional:
            c = self.data[split].conditions.astype(np.float32)
            x = (x, c)
        return x

    def y(self, split, output_channels=None):
        y = self.data[split].center_mpp
        if output_channels is not None:
            channel_ids = self.data[split].get_channel_ids(output_channels)
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

        output_types = []
        output_shapes = []

        # x
        x = self.x(split, is_conditional)
        if is_conditional:
            num = x[0].shape[0]
            output_types.append(tuple([tf.float32, tf.float32]))
            output_shapes.append(tuple([tf.TensorShape(x[0].shape[1:]), tf.TensorShape(x[1].shape[1:])]))
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