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
from miann.tl import Predictor
from miann.data import MPPData
from miann.utils import merged_config
import json
from copy import deepcopy
import scanpy as sc
from pynndescent import NNDescent
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

# annotation fns
def annotate_clustering(clustering, annotation, cluster_name, annotation_col=None):
    if annotation_col is None:
        return clustering
    if cluster_name == annotation_col:
        return clustering
    return np.array(annotation.set_index(cluster_name)[annotation_col].loc[clustering])

def add_clustering_to_adata(data_dir, cluster_name, adata, annotation, added_name=None, annotation_col=None):
    """
    adds cluster_name to adata.obs (as added_name) and cmap values for each cluster stored in cluster_name_annotation.csv to adata.uns
    uses annotation_col column if is not None

    name: name in adata.obs of added clustering
    annotation: annotation pandas dataframe
    """
    if added_name is None:
        added_name = annotation_col
        if added_name is None:
            added_name = cluster_name
    if annotation_col is None:
        annotation_col = cluster_name
    # load clustering
    clustering = np.load(os.path.join(data_dir, f'{cluster_name}.npy'), allow_pickle=True)
    # map clustering to annotation
    clustering = annotate_clustering(clustering, annotation, cluster_name, annotation_col)
    # add to adata
    adata.obs[added_name] = clustering
    adata.obs[added_name] = adata.obs[added_name].astype('category')
    # add cmap
    cmap = annotation.drop_duplicates(subset=annotation_col).set_index(annotation_col)[annotation_col+'_colors']
    adata.uns[added_name+"_colors"] = list(cmap.loc[adata.obs[added_name].cat.categories])


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
        # name of dataset that gives params for processing (except subsampling/subsetting)
        'process_like_dataset': None,
        'subsample': False, # either 'subsample' or 'som'
        'som_kwargs': {},
        'subsample_kwargs': {},
        'subset': False,
        'subset_kwargs': {},
        'seed': 42,
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
        # initialise cluster_mpp
        self._cluster_mpp = cluster_mpp
        # try to load it from disk if not already initialised
        self.cluster_mpp
        
        # initialise annotation
        self._cluster_annotation = None
        self.cluster_annotation

        # save config
        if save_config:
            config_fname = os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], 'cluster_params.json')
            os.makedirs(os.path.dirname(config_fname), exist_ok=True)
            json.dump(self.config, open(config_fname, 'w'), indent=4)
        
    @classmethod
    def from_cluster_data_dir(cls, data_dir):
        """
        data_dir: containing complete mpp_data with cluster_rep and cluster_name files. Relative to EXPERIMENT_DIR
        """
        # load mpp_data and cluster_params (for reference) from data_dir
        # TODO
        config_fname = os.path.join(EXPERIMENT_DIR, data_dir, 'cluster_params.json')
        config = json.load(open(config_fname, 'r'))
        return cls(config, save_config=False)

    @classmethod
    def from_exp(cls, exp:Experiment, cluster_config={}, data_dir=None):
        """
        Init from experiment for clustering of entire data that went into creating training data

        Args;
            exp: Experiment
            cluster_config: additional cluster params (like subsampling etc)
            data_dir: directory containing cluster_mpp (relative to exp.dir/exp.name)
        """
        add_cluster_config = cluster_config
        cluster_config = deepcopy(exp.config['cluster'])
        # add information on data to cluster
        cluster_config['data_config'] = exp.config['data']['data_config']
        cluster_config['data_dirs'] = exp.data_params['data_dirs']
        # only process data for model if experiment has model to evaluate
        cluster_config['process_like_dataset'] = exp.config['data']['dataset_name']
        cluster_config['seed'] = exp.data_params['seed']
        if data_dir is None:
            data_dir = os.path.join('aggregated', 'sub-'+cluster_config['subsample_kwargs']['frac'])
        cluster_config['cluster_data_dir'] = os.path.join(exp.dir, exp.name, data_dir)
        # add passed cluster_config
        cluster_config = merged_config(cluster_config, add_cluster_config)
        return cls(cluster_config, save_config=True)

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

    def set_cluster_name(self, cluster_name):
        """
        changes the cluster name and reloads cluster_mpp, and cluster_annotation
        """
        if self.config['cluster_name'] != cluster_name:
            self.config['cluster_name'] = cluster_name
            self._cluster_mpp = self._load_cluster_mpp()
            self._cluster_annotation = self._load_cluster_annotation()

    # --- Properties and loading fns --- 
    @property
    def cluster_mpp(self):
        """
        MPPData that is used for clustering.
        """
        if self._cluster_mpp is None:
            self._cluster_mpp = self._load_cluster_mpp()
        return self._cluster_mpp

    def _load_cluster_mpp(self, reload=False):
        """
        Load MPPData that is used for clustering.

        Tries to read MPPData with cluster_rep and cluster_name from cluster_data_dir. 

        Returns:
            MPPData or None if data could not be loaded
        """
        data_dir = self.config['cluster_data_dir']
        rep = self.config['cluster_rep']
        name = self.config['cluster_name']
        # check that dir is defined
        if data_dir is None:
            self.log.warn("Cluster_data_dir is None, cannot load cluster_mpp")
            return None
        # load data
        try:
            mpp_data = MPPData.from_data_dir(data_dir, base_dir=EXPERIMENT_DIR, optional_keys=['mpp', rep, name, 'umap'])
            self.log.info(f'Loaded cluster_mpp {mpp_data}')
            return mpp_data
        except FileNotFoundError as e:
            self.log.warn(f"Could not load MPPData from {data_dir}")
            return None

    @property
    def cluster_annotation(self):
        if self._cluster_annotation is None:
            self._cluster_annotation = self._load_cluster_annotation()
        return self._cluster_annotation
        
    def _load_cluster_annotation(self, recreate=False):
        """
        Read cluster annotation file / create it
        """
        fname = os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], f"{self.config['cluster_name']}_annotation.csv")
        # try to read file
        if os.path.exists(fname) and not recreate:
            annotation = pd.read_csv(fname, index_col=0, dtype=str, keep_default_na=False)
            return annotation
        else:
            # can create?
            if (self.cluster_mpp is None) or (self.cluster_mpp.data(self.config['cluster_name']) is None):
                self.log.info('cannot create annotation without clustering in cluster_mpp')
                return None
            # create empty annnotation from unique clusters and empty cluster for background in images
            annotation = pd.DataFrame({self.config['cluster_name']: sorted(list(np.unique(self.cluster_mpp.data(self.config['cluster_name']))), key=int)+[""]})
            annotation.index.name = 'index'
            # save annotation
            annotation.to_csv(fname)
            self._cluster_annotation = annotation
            # add colors
            self.add_cluster_colors(colors=None)
            return annotation

    # --- fns modifying annotation ---
    def add_cluster_annotation(self, annotation, to_col, from_col=None, colors=None):
        """
        add annotation and colormap to clustering. 
        
        Is saved in cluster_name_annotation.csv
        
        Args:
            annotation: dict with keys from "from_col" and values the annotation.
            to_col: name under which the annotation should be saved
            from_col: optionally set the annotation name from which to annotate. Default is cluster_name.
            colors: color dict, with annotation as keys and hex colors as values. Default is using tab20 colormap
        """
        if from_col is None:
            from_col = self.config['cluster_name']
        df = pd.DataFrame.from_dict(annotation, orient='index', columns=[to_col])
        # remove to_col and to_col_colors if present
        annotation = self.cluster_annotation
        annotation.drop(columns=[to_col], errors='ignore', inplace=True)
        # add annotation col
        annotation = pd.merge(annotation, df, how='left', left_on=from_col, right_index=True)
        self._cluster_annotation = annotation
        # save to disk
        fname = os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], f"{self.config['cluster_name']}_annotation.csv")
        self._cluster_annotation.to_csv(fname)
        # add colors
        self.add_cluster_colors(colors, to_col)
                                
    def add_cluster_colors(self, colors, from_col=None):
        """
        add colors to clustering or to annotation.
        
        adds column from_col"_colors" to self.cluster_annotation and saves it to cluster_name"_annotation.csv"
        
        Args:
            colors: color dict, with unique clustering values from from_col as keys and hex colors as values. 
                Default is using tab20 colormap
            from_col: optionally set clustering name for which to add colors. Default is cluster_name.
        """
        if from_col is None:
            from_col = self.config['cluster_name']
        to_col = from_col+'_colors'
        # remove previous colors
        annotation = self.cluster_annotation
        annotation.drop(columns=[to_col], errors='ignore', inplace=True)
        # add colors
        if colors is None:
            values = np.unique(annotation[from_col].dropna())
            N = len(values)
            cmap = plt.get_cmap('tab20', N)
            colors = {k:rgb2hex(cmap(i)) for i,k in enumerate(values)}
        df = pd.DataFrame.from_dict(colors, orient='index', columns=[to_col])
        annotation = pd.merge(annotation, df, how='left', left_on=from_col, right_index=True)
        # fill nan values with white background color
        annotation[to_col] = annotation[to_col].fillna('#ffffff')
        self._cluster_annotation = annotation
        # save to disk
        fname = os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], f"{self.config['cluster_name']}_annotation.csv")
        self._cluster_annotation.to_csv(fname)

    # --- getters ---
    def get_nndescent_index(self, recreate=False):
        index_fname =  os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir'], 'pynndescent_index.pickle')
        if os.path.isfile(index_fname) and not recreate:
            # load and return index
            return pickle.load(open(index_fname, 'rb'))
        # need to create index
        # check that cluster_rep has been computed already for cluster_mpp
        assert self.cluster_mpp.data(self.config['cluster_rep']) is not None
        self.log.info(f"Creating pynndescent index for {self.config['cluster_rep']}")

        data = self.cluster_mpp.data(self.config['cluster_rep'])
        if self.config['cluster_rep'] == 'mpp':
            data = self.cluster_mpp.center_mpp
        index = NNDescent(data.astype(np.float32))
        pickle.dump(index, open(index_fname, 'wb'))
        return index

    # --- functions creating and adding data to cluster_mpp ---
    def create_cluster_mpp(self):
        """
        Use cluster_params to create and save mpp_data to use for clustering

        Raises: ValueError if config does not contain data_dirs and process_like_dataset
        """
        # TODO: add option how to process data (e.g. for MPPcluster, do not need to add neighborhood)
        self.log.info('creating cluster mpp from config')
        # check that have required information
        if (len(self.config['data_dirs']) == 0):
            raise ValueError("Cannot create cluster data without data_dirs")
        self.log.info(f"processing cluster_mpp like dataset {self.config['process_like_dataset']}")
        # load params to use when processing
        data_config = get_data_config(self.config['data_config'])
        data_params = json.load(open(os.path.join(data_config.DATASET_DIR, self.config['process_like_dataset'], 'params.json'), 'r'))
        # load and process data
        mpp_data = []
        for data_dir in self.config['data_dirs']:
            mpp_data.append(MPPData.from_data_dir(data_dir, seed=self.config['seed'], data_config=self.config['data_config']))
            # subset if necessary (do before subsampling, to get expected # of samples)
            if data_params['subset']:
                mpp_data[-1].subset(**data_params['subset_kwargs'])
        mpp_data = MPPData.concat(mpp_data)
        # TODO after have reproduced data, could do subsampling inside for loop (after subsetting)
        if self.config['subsample']:
            mpp_data = mpp_data.subsample(add_neighborhood=data_params['neighborhood'], 
                neighborhood_size=data_params['neighborhood_size'],  **self.config['subsample_kwargs'])
        mpp_data.prepare(data_params)

        self._cluster_mpp = mpp_data
        if self.config['cluster_data_dir'] is not None:
            self._cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']))
  
    def predict_cluster_rep(self, exp):
        """
        Use exp to predict the necessary cluster representation
        """
        if self.cluster_mpp.data(self.config['cluster_rep']) is not None:
            self.log.info(f"cluster_mpp already contains key {self.config['cluster_rep']}. Not recalculating.")
            return
        pred = Predictor(exp)
        cluster_rep = pred.get_representation(self.cluster_mpp, rep=self.config['cluster_rep'])
        self.cluster_mpp._data[self.config['cluster_rep']] = cluster_rep
        # save cluster_rep
        if self.config['cluster_data_dir'] is not None:
            self.cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']), save_keys=[self.config['cluster_rep']])
        
    def create_clustering(self):
        """
        Cluster cluster_mpp using cluster_method defined in config.

        If cluster_data_dir is defined, saves clustering there.

        Raises: ValueError if cluster_rep is not available
        """
        # check that have cluster_rep
        if self.cluster_mpp.data(self.config['cluster_rep']) is None:
            raise ValueError(f"Key {self.config['cluster_rep']} is not available for clustering.")
        save_keys = [self.config['cluster_name']]
        # cluster 
        if self.config['cluster_method'] == 'leiden':
            self.log.info('Creating leiden clustering')
            # leiden clustering
            adata = self.cluster_mpp.get_adata(X=self.config['cluster_rep'])
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata, resolution=self.config['leiden_resolution'], key_added='clustering')
            self.cluster_mpp._data[self.config['cluster_name']] = np.array(adata.obs['clustering'])
            if self.config['umap']:
                self.log.info('Calculating umap')
                sc.tl.umap(adata)
                self.cluster_mpp._data['umap'] = adata.obsm['X_umap']
                save_keys.append('umap')
        elif self.config['cluster_method'] == 'kmeans':
            self.log.info('Creating kmeans clustering')
            from sklearn.cluster import KMeans
            est = KMeans(n_clusters=self.config['kmeans_n'], random_state=0)
            kmeans = est.fit(self.cluster_mpp.data(self.config['cluster_rep'])).labels_
            # TODO: cast kmeans to str?
            self.cluster_mpp._data[self.config['cluster_name']] = kmeans
        else:
            raise NotImplementedError(self.config['cluster_method'])
        
        # create and save pynndescent index
        _ = self.get_nndescent_index(recreate=True)
        
        # save to cluster_data_dir
        if self.config['cluster_data_dir'] is not None:
            self.cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']), 
                save_keys=save_keys)

        # add umap if not exists already
        if self.config['umap'] and not self.config['cluster_method'] == 'leiden':
            self.add_umap()

        # recreate annotation file
        self._cluster_annotation = self._load_cluster_annotation(recreate=True)

    def add_umap(self):
        """
        if umap does not yet exist, but should be calculated, calculates umap
        """
        if self.config['umap']:
            if self.cluster_mpp.data('umap') is not None:
                # already have umap, no need to calculate
                self.log.info('found existing umap, not recalculating.')
                return
            self.log.info("Calculating umap")
            adata = self.cluster_mpp.get_adata(X=self.config['cluster_rep'])
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            self.cluster_mpp._data['umap'] = adata.obsm['X_umap']
            # save to cluster_data_dir
            if self.config['cluster_data_dir'] is not None:
                self.cluster_mpp.write(os.path.join(EXPERIMENT_DIR, self.config['cluster_data_dir']), 
                    save_keys=['umap'])

    # --- using existing cluster_mpp, project clustering ---
    def project_clustering(self, mpp_data: MPPData, save_dir=None, batch_size=200000):
        """
        Project already computed clustering from cluster_mpp to mpp_data

        Args:
            mpp_data: MPPData to project the clustering to. Should contain cluster_rep
            save_dir: optional, full path to dir where the clustering should be saved
            batch_size: iterate over data in batches of size batch_size

        Returns:
            mpp_data with clustering
        """
        # check that clustering has been computed already for cluster_mpp
        assert self.cluster_mpp is not None
        assert self.cluster_mpp.data(self.config['cluster_name']) is not None
        assert mpp_data.data(self.config['cluster_rep']) is not None

        # get NNDescent index for fast projection
        index = self.get_nndescent_index()
        self.log.info(f"Projecting clustering to {len(mpp_data.obj_ids)} samples")
        # func for getting max count cluster in each row
        def most_frequent(arr):
            els, counts = np.unique(arr, return_counts=True)
            return els[np.argmax(counts)]
        # project clusters
        clustering = []
        samples = mpp_data.data(self.config['cluster_rep'])
        if self.config['cluster_rep'] == 'mpp':  # use center mpp in this special case
            samples = mpp_data.center_mpp
        for i in np.arange(0, samples.shape[0], batch_size):
            self.log.info(f'processing chunk {i}')
            cur_samples = samples[i:i+batch_size]
            neighs = index.query(cur_samples.astype(np.float32), k=15)[0]
            # NOTE: do not use apply_along_axis, because dtype is inferred incorrectly!
            clustering.append(np.array([most_frequent(row) for row in self.cluster_mpp.data(self.config['cluster_name'])[neighs]],
                dtype=self.cluster_mpp.data(self.config['cluster_name']).dtype))
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
        if exp.is_trainable:
            # create Predictor
            pred = Predictor(exp)
            # predict clustering on imgs
            img_save_dir = os.path.join(exp.full_path, f'results_epoch{pred.est.epoch:03d}', exp.config['evaluation']['split']+'_imgs')
            mpp_imgs = pred.est.ds.imgs[exp.config['evaluation']['split']]
            # add latent space to mpp_imgs + subset
            try:
                mpp_imgs.add_data_from_dir(img_save_dir, keys=[self.config['cluster_rep']], subset=True, base_dir='')
            except FileNotFoundError:
                self.log.warn(f"Did not find {self.config['cluster_rep']} in {img_save_dir}. Run create_clustering first.")
                return 
        else:
            img_save_dir = os.path.join(exp.full_path, 'results_epoch000', exp.config['evaluation']['split']+'_imgs')
            mpp_imgs = MPPData.from_data_dir(img_save_dir, base_dir='')
        self.log.info(f'Projecting cluster_imgs for {exp.dir}/{exp.name}')
        return self.project_clustering(mpp_imgs, save_dir=img_save_dir)