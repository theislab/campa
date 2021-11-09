import os
from miann.constants import EXPERIMENT_DIR
from miann.utils import merged_config
from miann.tl import LossEnum, ModelEnum
import json
from copy import deepcopy
import logging
import pandas as pd
import re
import tensorflow as tf
from miann.data import MPPData

class Experiment:
    # base experiment config
    config = {
        'experiment': {
            'dir': None,
            'name': 'experiment',
            'save_config': True,
        },
        'data': { 
            'data_config': None,
            'dataset_name': None,
            'output_channels': None,
        },
        'model': {
            'model_cls': ModelEnum.BaseAEModel, # instance or value of ModelEnum
            'model_kwargs': {},
            # if true, looks for saved weights in experiment_dir
            # if a path, loads these weights
            'init_with_weights': False,
        },
        'training': {
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 128,
            'loss': {'decoder':LossEnum.MSE}, # instance or value of LossEnum
            'loss_weights': {'decoder': 1},
            'loss_warmup_to_epoch': {},
            'metrics': {'decoder': LossEnum.MSE}, # instance or value of LossEnum
            # saving models
            'save_model_weights': True,
            'save_history': True,
            'overwrite_history': True,
        },
        'evaluation': { # TODO change this to fit to aggregation params
            'split': 'val',
            'predict_reps': ['latent', 'decoder'],
            'img_ids': 25,
            'predict_imgs': True
        },
        'cluster': {  # cluster config, also used in this format for whole data clustering
            'predict_cluster_imgs': True,
            'cluster_name': 'clustering',
            'cluster_rep': 'latent',
            'cluster_method': 'leiden', # leiden or kmeans
            'leiden_resolution': 0.8,
            'subsample': None, # 'subsample' or 'som'
            'subsample_kwargs': {},
            'som_kwargs': {},
            'umap': True,
        },
    }
    
    def __init__(self, config):
        self.config = merged_config(self.config, config)
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info(f'Setting up experiment {self.dir}/{self.name}')
        # create exp_path
        if self.dir is not None:
            os.makedirs(self.full_path, exist_ok=True)
            if self.config['experiment']['save_config']:
                self.log.info(f'Saving config to {self.dir}/{self.name}/config.json')
                json.dump(self.config, open(os.path.join(self.full_path, 'config.json'), "w"), indent=4)
        else:
            self.log.info('exp_dir is None, did not save config')


    def from_dir(cls, exp_path):
        """
        init experiment from trained experiment in exp_path.
        Changes init_with_weights to True and save_config to False
        """
        # load config from json
        config_fname = os.path.join(EXPERIMENT_DIR, exp_path, 'config.json')
        assert os.path.exists(config_fname), "no config.json in {}".format(exp_path)
        config = json.load(open(config_fname))
        # set save_config to False to avoid overwriting
        config['experiment']['save_config'] = False
        self = cls(config)
        self.log.info(f'Initialised from existing experiment in {self.dir}/{self.name}')

    def set_to_evaluate(self):
        # changes init_with_weights to True to load correct weights in Estimator
        self.config['model']['init_with_weights'] = True
        return self

    @property
    def name(self):
        return self.config['experiment']['name']

    @property
    def dir(self):
        return self.config['experiment']['dir']

    @property
    def full_path(self):
        return os.path.join(EXPERIMENT_DIR, self.dir, self.name)

    @property
    def estimator_config(self):
        estimator_config = {key:val for key,val in self.config.items() if key in ['experiment', 'data', 'model', 'training']}
        # return copy to avoid side effects on self.config
        return deepcopy(estimator_config)

    @property
    def evaluate_config(self):
        evaluate_config = self.config['evaluation']
        return deepcopy(evaluate_config)


    def get_history(self):
        history_path = os.path.join(self.full_path, 'history.csv')
        if os.path.isfile(history_path):
            return pd.read_csv(self.history_name, index_col=0)
        else:
            return None

    @property
    def epoch(self):
        """
        last epoch for which there is a trained model
        """
        weights_path = tf.train.latest_checkpoint(self.full_path)
        if weights_path is None:
            return 0
        # find epoch in weights_path
        res = re.findall(r'epoch(\d\d\d)', os.path.basename(weights_path))
        if len(res) == 0:
            return 0
        else:
            return int(res[0])

    # TODO test in ModelComparator etc!
    def get_split_mpp_data(self):
        """
        val / test from results_epochXXX
        """
        split = self.config['evaluation']['split']
        data_dir = os.path.join(self.full_path, f'results_epoch{self.epoch:03d}', split)
        if os.path.isdir(data_dir):
            return MPPData.from_data_dir(data_dir, base_dir='', keys=['x', 'y', 'obj_ids', 'mpp'], 
            optional_keys=list(set([self.config['cluster_rep'], 'latent', 'decoder', self.config['cluster_name']])), 
            data_config=self.config['data']['data_config'])
        return None

    def get_split_imgs_mpp_data(self):
        """
        val_imgs / test_imgs from results_epochXXX
        """
        split = self.config['evaluation']['split']
        data_dir = os.path.join(self.full_path, f'results_epoch{self.epoch:03d}', split+'_imgs')
        if os.path.isdir(data_dir):
            return MPPData.from_data_dir(data_dir, base_dir='', keys=['x', 'y', 'obj_ids', 'mpp'], 
            optional_keys=list(set([self.config['cluster_rep'], 'latent', 'decoder', self.config['cluster_name']])), 
            data_config=self.config['data']['data_config'])
        return None

    def get_sub_mpp_data(self):
        """
        subsampled mpp data from aggregated/sub
        """
        pass
