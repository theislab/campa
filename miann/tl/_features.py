
from miann.tl import Experiment
from miann.data import MPPData
from miann.constants import get_data_config
import os
import numpy as np
import pandas as pd
import anndata as ad
import logging
from copy import deepcopy

class FeatureExtractor:
    """
    Extract features from clustering.
    """

    def __init__(self, exp, data_dir, cluster_name, cluster_dir=None, cluster_col=None, adata=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        cluster_col = cluster_col if cluster_col is not None else cluster_name
        self.params = {
            "data_dir": data_dir,
            "cluster_name": cluster_name,
            "cluster_dir": cluster_dir,
            "cluster_col": cluster_col,
            "exp_name": exp.dir + '/' + exp.name
        }
        self.adata = adata

        self.annotation = self.exp.get_cluster_annotation(self.params['cluster_name'], self.params['cluster_dir'])
        clusters = list(np.unique(self.annotation[self.params['cluster_col']]))
        clusters.remove('')
        self.clusters = clusters

        self._mpp_data = None
        
    @property
    def mpp_data(self):
        if self._mpp_data is None:
            self._mpp_data = MPPData.from_data_dir(self.params['data_dir'], base_dir=os.path.join(self.exp.full_path, 'aggregated/full_data'), 
            keys=['x', 'y', 'mpp', 'obj_ids', self.params['cluster_name']])
            # ensure that cluster data is string
            self._mpp_data._data[self.params['cluster_name']] = self._mpp_data._data[self.params['cluster_name']].astype(str)
            # prepare according to data_params
            data_params = deepcopy(self.exp.data_params)
            self._mpp_data.prepare(data_params)
        return self._mpp_data

    @classmethod
    def from_adata(cls, fname):
        """
        Initialise from existing adata

        Args;
            fname: full path to adata object
        """
        adata = ad.read(fname)
        params = adata.uns['params']
        exp = Experiment.from_dir(params.pop('exp_name'))
        self = cls(exp, adata=adata, **params)
        self.fname = fname
        return self

    def extract_intensity_size(self, force=False, fname="features.h5ad"):
        """
        calculate per cluster mean intensity and size for each object.

        Saves adata in exp.full_path/aggregated/full_data/data_dir/features.h5ad

        Args:
            force: overwrite existing adata
        """
        if self.adata is not None and not force:
            self.log.info('extract_intensity_size: adata is not None. Specify force=True to overwrite. Exiting.')
        self.log.info(f"Calculating {self.params['cluster_name']} (col: {self.params['cluster_col']}) mean and size for {self.params['data_dir']}")
        df = pd.DataFrame(data=self.mpp_data.center_mpp, columns=list(self.mpp_data.channels.name), 
                             index=self.mpp_data.obj_ids)
        # create adata with X = mean intensity of "all" cluster
        grouped = df.groupby(df.index)
        adata = ad.AnnData(X=grouped.mean())

        # add all metadata
        OBJ_ID = self.mpp_data.data_config.OBJ_ID
        metadata = self.mpp_data.metadata.copy()
        metadata[OBJ_ID] = metadata[OBJ_ID].astype(str)
        metadata = pd.merge(metadata, adata.obs, how='right', left_on=OBJ_ID, right_index=True)
        metadata = metadata.reset_index(drop=True)  # make new index, keep mapobject_id in column
        metadata.index = metadata.index.astype(str) # make index str, because adata does not play nice with int indices
        # add int col of mapobject_id for easier merging
        metadata['obj_id_int'] = metadata[OBJ_ID].astype(int)
        adata.obs = metadata 

        # add size of all cluster
        # reindex to ensure all object are present
        size = grouped[list(df.columns)[0]].count().reindex(adata.obs['obj_id_int'])
        adata.obsm['size'] = pd.DataFrame(columns=['all']+self.clusters, index=adata.obs.index)
        adata.obsm['size']['all'] = np.array(size)

        # add uns metadata
        adata.uns['clusters'] = self.clusters
        adata.uns['params'] = self.params

        # add intensities of each cluster as a layer in adata and fill size obsm
        for c in self.clusters:
            self.log.debug(f'processing {c}')
            # get cluster ids to mask
            c_ids = list(self.annotation[self.annotation[self.params['cluster_col']] == c][self.params['cluster_name']])
            mask = np.where(np.isin(self.mpp_data.data(self.params['cluster_name']), c_ids))
            cur_df = df.iloc[mask]
            # group by obj_id  
            grouped = cur_df.groupby(cur_df.index)
            # add mean of cluster
            # reindex to ensure all object are present
            mean = grouped.mean().reindex(adata.obs['obj_id_int'])
            mean = mean.fillna(0)
            adata.layers[f'intensity_{c}'] = np.array(mean[adata.var.index])
            # add size
            # reindex to ensure all object are present
            size = grouped[list(df.columns)[0]].count().reindex(adata.obs['obj_id_int'])
            adata.obsm['size'][c] = np.array(size)
        # fill nans in size obsm
        adata.obsm['size'] = adata.obsm['size'].fillna(0)

        self.adata = adata

        # write to disk
        fname = os.path.join(self.exp.full_path, "aggregated/full_data", self.params['data_dir'], fname)
        self.log.info(f'saving adata to {fname}')
        self.fname = fname
        self.adata.write(self.fname)

    def get_intensity_adata(self):
        """
        adata object with intensity per cluster combined in X. Needed for intensity and dotplots.
        """
        adata = self.adata
        adatas = {}
        cur_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
        cur_adata.obs['size'] = adata.obsm['size']['all']
        adatas['all'] = cur_adata
        for c in adata.uns['clusters']:
            cur_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
            cur_adata.X = adata.layers[f'intensity_{c}']
            cur_adata.obs['size'] = adata.obsm['size'][c]
            adatas[c] = cur_adata
        comb_adata = ad.concat(adatas, uns_merge='same', index_unique='-', label='cluster')
        return comb_adata

    def extract_intensity_csv(self, obs=None):
        """
        extract csv file containing obj_id, mean cluster intensity and size for each channel.

        saves csv as fname.csv
        obs: column names from metadata.csv that should be additionally stored.
        """
        if self.adata is None:
            self.log.warn("Intensity and size information is not present. Calculate extract_intensity_size first! Exiting.")
            return
        adata = self.get_intensity_adata()
        df = pd.DataFrame(data=adata.X, columns=adata.var_names)
        # add size
        df['size'] = np.array(adata.obs['size'])
        # add cluster and obj_id
        OBJ_ID = get_data_config(self.exp.config['data']['data_config']).OBJ_ID
        df['cluster'] = np.array(adata.obs['cluster'])
        df[OBJ_ID] = np.array(adata.obs[OBJ_ID])
        # add additional obs
        for col in obs:
            df[col] = np.array(adata.obs[col])
        # save csv
        df.to_csv(os.path.splitext(self.fname)[0]+'.csv')