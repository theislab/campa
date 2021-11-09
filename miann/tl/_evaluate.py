from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from miann.tl import Experiment

# TODO working on evaluation + clustering pipeline for exp model. 
# TODO when finished: create fns for how to aggregate - should easily be possible with this code!
import numpy as np
import os
import tensorflow as tf
import logging
from miann.constants import EXPERIMENT_DIR, get_data_config
from miann.tl import Estimator
from miann.data import MPPData
from miann.utils import merged_config
import json
from copy import deepcopy
import scanpy as sc
from pynndescent import NNDescent
import pickle

# TODO aggregator should use Cluster to get the clustering on all data
# TODO aggregator deals with multiple data dirs etc
class Cluster:
    """
    cluster MPPData.

    Has functions to create a (subsampled) MPPData for clustering, cluster_mpp, cluster it, 
    and to project the clustering to other MPPDatas.
    """
    config = {
        # --- cluster data creation (mpp_data is saved to cluster_data_dir) ---
        'data_config': 'NascentRNA',
        'data_dirs': [],
        'add_data_dirs': None,
        # name of dataset that gives params for processing (except subsampling/subsetting)
        'process_like_dataset': None,
        'subsample': False, # either 'subsample' or 'som'
        'som_kargs': {},
        'subsample_kwargs': {},
        'subset': False,
        'subset_kwargs': {},
        # name of the dir containing the mpp_data that is clustered. Relative to EXPERIMENT_DIR
        'cluster_data_dir': None,
        # --- cluster params ---
        # name of the cluster assignment file
        'cluster_name': 'clustering',
        # representation that should be clustered (name of existing file)
        'cluster_rep': 'latent',
        'cluster_method': 'leiden', # leiden or kmeans
        'leiden_resolution': 0.8,
        'kmeans_n': 20,
        'umap': True, # calculate umap of cluster data
    }

    def __init__(self, config, cluster_mpp=None, save_config=False):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = merged_config(self.config, config)
        self.data_config = get_data_config(self.config['data_config'])
        # load dataset_params
        self.dataset_params = None
        if self.config['process_like_dataset'] is not None:
            params_fname = os.path.join(self.data_config.DATASET_DIR, self.config['process_like_dataset'], 'params.json')
            self.dataset_params = json.load(open(params_fname, 'r'))
        self._cluster_mpp = cluster_mpp
        # try to load cluster_mpp from disk to ensure that is properly initialised
        self.get_cluster_mpp()

        # save config
        if save_config:
            config_fname = os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], 'cluster_params.json')
            json.dump(self.config, open(config_fname, 'w'), indent=4)
        
    @classmethod
    def from_cluster_data_dir(cls, data_dir):
        """
        data_dir: containing complete mpp_data with cluster_rep and cluster_name files
        """
        # load mpp_data and cluster_params (for reference) from data_dir
        # TODO
        config_fname = os.path.join(EXPERIMENT_DIR, data_dir, 'cluster_params.json')
        config = json.load(open(config_fname, 'r'))
        return cls(config, save_config=False)

    @classmethod
    def from_exp_split(cls, exp: Experiment):
        """
        Init from experiment for clustering of val/test split
        """
        # TODO load exp 
        #est = Estimator(exp.set_to_evaluate())
        #mpp_data = est.ds.data[exp.evaluate_config['split']]
    
        cluster_config = deepcopy(exp.config['cluster'])
        # add data_config
        cluster_config['data_config'] = exp.config['data']['data_config']
        cluster_config['subsample'] = False
        cluster_config['cluster_data_dir'] = os.path.join(exp.dir, exp.name, f'results_epoch{exp.epoch:03d}', exp.evaluate_config['split'])

        return cls(cluster_config, save_config=True)

    def get_cluster_mpp(self, reload=False):
        """
        MPPData that is used for clustering.

        If not initialised, tries to read it from cluster_data_dir. 
        In addition, tries to read cluster_rep and cluster_name from cluster_data_dir and add it to 
        an existing cluster_mpp

        Returns:
            MPPData or None if data could not be loaded
        """
        data_dir = self.config['cluster_data_dir']
        rep = self.config['cluster_rep']
        name = self.config['cluster_name']
        # check that dir is defined
        if data_dir is None:
            self.log.warn("Cluster_data_dir is None, cannot load cluster_mpp")
            return self._cluster_mpp
        # load data
        if (self._cluster_mpp is None) or reload:
            try:
                mpp_data = MPPData.from_data_dir(data_dir, base_dir=EXPERIMENT_DIR, optional_keys=[rep, name])
                self.log.info(f'Loaded cluster_mpp {mpp_data}')
                self._cluster_mpp = mpp_data
            except FileNotFoundError as e:
                self.log.warn(f"Could not load MPPData from {data_dir}")
                raise e
                return None
        return self._cluster_mpp


    def get_nndescent_index(self, recreate=False):
        index_fname =  os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], 'pynndescent_index.pickle')
        if os.path.isfile(index_fname) and not recreate:
            # load and return index
            return pickle.load(open(index_fname, 'rb'))
        # need to create index
        cluster_mpp = self.get_cluster_mpp()
        # check that cluster_rep has been computed already for cluster_mpp
        assert cluster_mpp.data(self.config['cluster_rep']) is not None
        self.log.info(f"Creating pynndescent index for {self.config['cluster_rep']}")
        index = NNDescent(cluster_mpp.data(self.config['cluster_rep']).astype(np.float32))
        pickle.dump(index, open(index_fname, 'wb'))
        return index

    def create_cluster_data(self):
        """
        Use cluster_params to create and save mpp_data to use for clustering

        Raises: ValueError if config does not contain data_dirs and process_like_dataset
        """
        # check that have required information
        if (len(self.config['data_dirs']) == 0) or self.config['process_like_dataset'] is None:
            raise ValueError("Cannot create cluster data without data_dirs and process_like_dataset")
        if self.config['add_data_dirs'] is not None:
            assert len(self.config['data_dirs']) == len(self.config['add_data_dirs'])

        # TODO load mpp data + subsample (or SOM)
        # TODO save resulting mpp data to cluster_data_dir
              # 1. subset mpp_data
        #if cluster_params['subsample'] == 'subsample':
        #    mpp_data = mpp_data.subsample(**cluster_params['subsample_kwargs'])
        #    save_only_clustering = False
        #elif cluster_params['subsample'] == 'som':
        #    raise NotImplementedError('som') # TODO copy + adapt SOM code
        #    save_only_clustering = False
        #elif cluster_params['subsample'] is None:
        #    pass
        #else:
        #    raise NotImplementedError(cluster_params['subsample'])
        # load cluster_mpp
        return self.get_cluster_mpp()

    def predict_cluster_rep(self, exp):
        """
        Use exp to predict the necessary cluster representation
        """
        cluster_mpp = self.get_cluster_mpp()
        if cluster_mpp.data(self.config['cluster_rep']) is not None:
            self.log.info(f"cluster_mpp already contains key {self.config['cluster_rep']}. Not recalculating.")
            return
        pred = Predictor(exp)
        cluster_rep = pred.get_representation(cluster_mpp, rep=self.config['cluster_rep'])
        cluster_mpp._data[self.config['cluster_rep']] = cluster_rep
        # save cluster_rep
        if self.config['cluster_data_dir'] is not None:
            cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']), save_keys=[self.config['cluster_rep']])
        
    def create_clustering(self):
        """
        Cluster cluster_mpp using cluster_method defined in config.

        If cluster_data_dir is defined, saves clustering there.

        Raises: ValueError if cluster_rep is not available
        """
        cluster_mpp = self.get_cluster_mpp()
        # check that have cluster_rep
        if cluster_mpp.data(self.config['cluster_rep']) is None:
            raise ValueError(f"Key {self.config['cluster_rep']} is not available for clustering.")
        # cluster 
        if self.config['cluster_method'] == 'leiden':
            self.log.info('Creating leiden clustering')
            # leiden clustering
            adata = cluster_mpp.get_adata(X=self.config['cluster_rep'])
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata, resolution=self.config['leiden_resolution'], key_added='clustering')
            cluster_mpp._data[self.config['cluster_name']] = np.array(adata.obs['clustering'])
            if self.config['umap']:
                self.log.info('Calculating umap')
                sc.tl.umap(adata)
                cluster_mpp._data['umap'] = adata.obsm['X_umap']
        elif self.config['cluster_method'] == 'kmeans':
            self.log.info('Creating kmeans clustering')
            from sklearn.cluster import KMeans
            est = KMeans(n_clusters=self.config['kmeans_n'], random_state=0)
            kmeans = est.fit(cluster_mpp.data(self.config['cluster_rep'])).labels_
            # TODO: cast kmeans to str?
            cluster_mpp._data[self.config['cluster_name']] = kmeans
        else:
            raise NotImplementedError(self.config['cluster_method'])
        
        # create and save pynndescent index
        _ = self.get_nndescent_index(recreate=True)
        
        # add umap if not exists already
        if self.config['umap'] and not self.config['cluster_method'] == 'leiden':
            self.log.info("Calculating umap")
            adata = cluster_mpp.get_adata(X=cluster_mpp['cluster_rep'])
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            cluster_mpp._data['umap'] = adata.obsm['X_umap']

        self._cluster_mpp = cluster_mpp
        # save to cluster_data_dir
        if self.config['cluster_data_dir'] is not None:
            cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']), 
                save_keys=[self.config['cluster_name'], 'umap'])

    def predict_clustering(self, mpp_data: MPPData, save_dir=None, batch_size=200000):
        """
        Project already computed clustering from cluster_mpp to mpp_data

        Args:
            mpp_data: MPPData to project the clustering to. Should contain cluster_rep
            save_dir: optional, full path to dir where the clustering should be saved
            batch_size: iterate over data in batches of size batch_size

        Returns:
            mpp_data with clustering
        """
        cluster_mpp = self.get_cluster_mpp()
        # check that clustering has been computed already for cluster_mpp
        assert cluster_mpp is not None
        assert cluster_mpp.data(self.config['cluster_name']) is not None
        assert mpp_data.data(self.config['cluster_rep']) is not None

        # get NNDescent index for fast projection
        index = self.get_nndescent_index()
        self.log.info("Projecting clustering to {len(mpp_data.x)} sampled")
        # func for getting max count cluster in each row
        def most_frequent(arr):
            els, counts = np.unique(arr, return_counts=True)
            return els[np.argmax(counts)]
        # project clusters
        clustering = []
        samples = mpp_data.data(self.config['cluster_rep'])
        for i in np.arange(0, samples.shape[0], batch_size):
            self.log.info(f'processing chunk {i}')
            cur_samples = samples[i:i+batch_size]
            neighs = index.query(cur_samples.astype(np.float32), k=15)[0]
            clustering.append(np.apply_along_axis(most_frequent, 
                arr=cluster_mpp.data(self.config['cluster_name'])[neighs], axis=1))
        clustering = np.concatenate(clustering)
        mpp_data._data[self.config['cluster_name']] = clustering

        # save
        if save_dir is not None:
            mpp_data.write(save_dir, save_keys=[self.config['cluster_name']])
        return mpp_data

    def predict_cluster_imgs(self, exp):
        """
        Predict cluster imgs from experiment
        """
        # create Predictor
        pred = Predictor(exp)
        # predict clustering on imgs
        img_save_dir = os.path.join(EXPERIMENT_DIR, exp.dir, exp.name, f'results_epoch{pred.est.epoch:03d}', exp.config['evaluation']['split']+'_imgs')
        mpp_imgs = pred.est.ds.imgs[exp.config['evaluation']['split']]
        # add latent space to mpp_imgs + subset
        try:
            mpp_imgs.add_data_from_dir(img_save_dir, keys=[self.config['cluster_rep']], subset=True, base_dir='')
        except FileNotFoundError:
            self.log.warn(f"Did not find {self.config['cluster_rep']} in {img_save_dir}. Run create_clustering first.")
            return 
        self.log.info(f'Projecting cluster_imgs for {exp.dir}/{exp.name}')
        return self.predict_clustering(mpp_imgs, save_dir=img_save_dir)

class Predictor:
    """
    predict MPPData with trained model.    
    """

    def __init__(self, exp, batch_size=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp.set_to_evaluate()
        self.log.info(f'Creating Predictor for {self.exp.dir}/{self.exp.name}')
        # set batch size
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = self.exp.config['evaluation'].get('batch_size', self.exp.config['training']['batch_size'])
        
        # build estimator
        self.est = Estimator(exp)

    def evaluate_model(self):
        """
        predict val/test split and imgs.

        Uses exp.evaluate_config for settings
        """
        config = self.exp.evaluate_config
        # predict split
        self.predict_split(config['split'], reps=config['predict_reps'])
        if config['predict_imgs']:
            self.predict_split(config['split']+'_imgs', img_ids=config['img_ids'], reps=config['predict_reps'])

    def predict(self, mpp_data, save_dir=None, reps=['latent'], mpp_params={}):
        """
        Predict reps from mpp_data, 
        
        Args:
        save_dir: save predicted reps to this dir
        reps: which representations to predict
        mpp_params: base_data_dir and subset information to save alongside of predicted mpp_data. See MPPData.write()

        Returns:
            MPPData with keys reps.
        """
        for rep in reps:
            mpp_data._data[rep] = self.get_representation(mpp_data, rep=rep)
        if save_dir is not None:
            mpp_data.write(save_dir, save_keys=reps, mpp_params=mpp_params)
        return mpp_data

    def predict_split(self, split, img_ids=None, reps=['latent', 'decoder'], **kwargs):
        """
        Predict data from train/val/test split of dataset that the model was trained with.
        Saves in experiment_dir/exp_name/split. 
        
        Args:
            split (str or list of str): train, val, test, val_imgs, test_imgs
            img_ids: obj_ids or number of objects that should be predicted (only for val_imgs and test_imgs)
            reps: which representations should be predicted?
        """
        self.log.info(f'Predicting split {split} for {self.exp.dir}/{self.exp.name}')
        if '_imgs' in split:
            mpp_data = self.est.ds.imgs[split.replace('_imgs', '')]
            if img_ids is None:
                img_ids = list(mpp_data.unique_obj_ids)
            if isinstance(img_ids, int):
                # choose random img_ids from available ones
                rng = np.random.default_rng(seed=42)
                img_ids = rng.choice(mpp_data.unique_obj_ids, img_ids, replace=False)
            # subset mpp_data to these img_ids
            mpp_data.subset(obj_ids=img_ids)
            # add neighborhood to mpp_data (other processing is already done)
            if self.est.ds.params['neighborhood']:
                mpp_data.add_neighborhood(size=self.est.ds.params['neighborhood_size'])
        else:
            mpp_data = self.est.ds.data[split]

        for rep in reps:
            mpp_data._data[rep] = self.get_representation(mpp_data, rep=rep)
        save_dir = os.path.join(self.exp.full_path, f'results_epoch{self.est.epoch:03d}', split)
        # base data dir for correct recreation of mpp_data
        base_data_dir = os.path.join("datasets", self.est.ds.params['dataset_name'], split)
        mpp_data.write(save_dir, save_keys=reps, mpp_params={'base_data_dir':base_data_dir, 'subset': True})

    def get_representation(self, mpp_data, rep='latent'):
        """
        Return desired representation from given mpp_data inputs.

        TODO might remove entangled, latent_y in the future (not needed currently)

        Args:
            rep: Representation, one of: 'input', 'latent', 'entangled' (for cVAE models which have and entangled layer in the decoder),
                'decoder', 'latent_y' (encoder_y)
        Returns:
            representation
        """
        if rep == 'input':
            # return mpp
            return mpp_data.center_mpp
        # need to prepare input to model
        if self.est.model.is_conditional:
            data = [mpp_data.mpp, mpp_data.conditions]
        else:
            data = mpp_data.mpp
        # get representations
        if rep == 'latent':
            return self.est.model.encoder.predict(data, batch_size=self.batch_size)
        elif rep == 'entangled':
            # this is only for cVAE models which have an "entangled" layer in the decoder
            # create the model for predicting the latent
            decoder_to_entangled_latent = tf.keras.Model(self.est.model.decoder.input, self.est.model.entangled_latent)
            encoder_to_entangled_latent = tf.keras.Model(self.est.model.input,
                                                         decoder_to_entangled_latent([self.est.model.encoder(self.est.model.input), self.est.model.input[1]]))
            return encoder_to_entangled_latent.predict(data, batch_size=self.batch_size)
        elif rep == 'decoder':
            return self.est.predict_model(data, batch_size=self.batch_size)
        elif rep == 'latent_y':
            return self.est.model.encoder_y.predict(data, batch_size=self.batch_size)
        else:
            raise NotImplementedError(rep)
