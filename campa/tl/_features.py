
from campa.tl import Experiment
from campa.data import MPPData
from campa.constants import get_data_config, CoOccAlgo, CO_OCC_CHUNK_SIZE
import os
import numpy as np
import pandas as pd
import anndata as ad
import logging
from copy import deepcopy
import squidpy as sq
from campa.tl._cluster import annotate_clustering
from typing import Union
import time
import multiprocessing
from functools import partial
import tqdm
from skimage.measure import label, regionprops

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
            return
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

    def extract_object_stats(self, area_threshold=0):
        """
        Extract features from connected components per cluster for each cell.
        
        Extracts number, area, circlularity, elongation, and extent of connected components per cluster for each cell.
        For every feature except count the mean, std, and median of this feature per cell is calculated.

        Adds obsm entries: object_count, object_area_{mean|std|median}, object_circularity_{mean|std|median}, 
            object_elongation_{mean|std|median}, object_extent_{mean|std|median}


        object_circularity_{mean|std|median} = (4 * pi * Area) / Perimeter^2, 
        object_elongation_{mean|std|median} = (major_axis - minor_axis) / major_axis

        Args:
            area_threshold: all components smaller than this threshold are discarded
        """
        if self.adata is None:
            self.log.info('extract_object_stats: adata is None. Calculate it with extract_intensity_size before extracting object stats. Exiting.')
            return
        self.log.info(f"calculating object stats with area threshold {area_threshold} for clustering {self.params['cluster_name']} (col: {self.params['cluster_col']})")
        cluster_names = {n: i for i,n in enumerate(self.clusters + [''])}
        
        counts = []
        features = {
            feature: {agg: [] for agg in ['mean', 'std', 'median']} for feature in ['area', 'circularity', 'elongation', 'extent']
        }
        obj_ids = []
        for obj_id in self.mpp_data.unique_obj_ids:
            mpp_data = self.mpp_data.subset(obj_ids=[obj_id], copy=True)
            img, (pad_x, pad_y) = mpp_data.get_object_img(obj_id, data=self.params['cluster_name'], annotation_kwargs={'annotation': self.annotation, 'to_col': self.params['cluster_col']})
            # convert labels to numbers
            img = np.vectorize(cluster_names.__getitem__)(img[:,:,0])
            label_img = label(img, background=len(self.clusters), connectivity=2)
            # iterate over all regions in this image
            obj_counts = np.zeros(len(self.clusters))
            obj_features = {feature: [[] for _ in self.clusters] for feature in ['area', 'circularity', 'elongation', 'extent']}
            for region in regionprops(label_img, intensity_image=img):
                if region.area > area_threshold:
                    assert region.min_intensity == region.max_intensity
                    c = region.min_intensity
                    obj_counts[c] += 1
                    obj_features['area'][c].append(region.area)
                    # circularity can max be 1, larger values are due to tiny regions where perimeter is overestimated
                    obj_features['circularity'][c].append(min(4*np.pi*region.area/(region.perimeter**2), 1))
                    obj_features['elongation'][c].append((region.major_axis_length - region.minor_axis_length) / region.major_axis_length)
                    obj_features['extent'][c].append(region.extent)
            counts.append(obj_counts)
            for feature in features.keys():
                for agg in ['mean', 'std', 'median']:
                    agg_feature = [eval(f'np.{agg}')(f, axis=0) for f in obj_features[feature]]
                    features[feature][agg].append(agg_feature)
            obj_ids.append(obj_id)
        features = {f'{feature}_{agg}': features[feature][agg] for feature in features.keys() for agg in ['mean', 'std', 'median']}
        features['count'] = counts

        # save to adata
        for name, res in features.items():
            df = pd.DataFrame(np.array(res), index=obj_ids, columns=self.clusters)
            df.index = df.index.astype(str)
            # ensure obj_ids are in correct order
            df = pd.merge(df, self.adata.obs, how='right', left_index=True, right_on='mapobject_id', suffixes=('','right'))[df.columns]
            df = df.fillna(0)
            # add to adata.obsm
            self.adata.obsm['object_'+name] = df
        self.adata.uns['object_stats_params'] = {'area_threshold': area_threshold}
        # write adata
        self.log.info(f'saving adata to {self.fname}')
        self.log.info(f'adata params {self.adata.uns["params"]}')
        self.adata.uns['params'] = self.params # add params to adata again, because exp_name keeps getting lost for some reason (this is a really weird bug...)
        self.adata.write(self.fname)


    def extract_co_occurrence(self, interval, algorithm: Union[str,CoOccAlgo] = CoOccAlgo.OPT, num_processes = None):
        """
        Extract co_occurrence for each cell invididually. 

        Adds obsm co_occurrence_CLUSTER1_CLUSTER2 to adata and saves to self.fname

        Args:
            interval: distance intervals for which to calculate co-occurrence score
            algorithm: co-occurrence function to use. 
                squidpy: use sq.gr.co_occurrence
                opt: use custom implementation which is optimised for a large number of pixels. 
                    This implementation avoids recalculation of distances, using the fact that coordinates
                    in given images lie on a regular grid.
                Use opt for very large inputs
            num_processes: only for algorithm='opt'. Number if processes to use to compute scores.
        """
        if self.adata is None:
            self.log.info('extract_co_occurrence: adata is None. Calculate it with extract_intensity_size before extracting co_occurrence. Exiting.')
            return
        self.log.info(f"calculating co-occurrence for intervals {interval} and clustering {self.params['cluster_name']} (col: {self.params['cluster_col']})")
        if CoOccAlgo(algorithm) == CoOccAlgo.OPT:
            cluster_names = {n: i for i,n in enumerate(self.clusters + [''])}
            coords2_list = _prepare_co_occ(interval)
        elif CoOccAlgo(algorithm) == CoOccAlgo.SQUIDPY:
            cluster_names = {n: i for i,n in enumerate(self.clusters)}
        
        obj_ids = []
        co_occs = []
        chunks = 20
        i=0
        missing_obj_ids = self._missing_co_occ_obj_ids()
        self.log.info(f'calculating co-occurrence for {len(missing_obj_ids)} objects')
        for obj_id in missing_obj_ids:
            if CoOccAlgo(algorithm) == CoOccAlgo.OPT:
                mpp_data = self.mpp_data.subset(obj_ids=[obj_id], copy=True)
                img, (pad_x, pad_y) = mpp_data.get_object_img(obj_id, data=self.params['cluster_name'], annotation_kwargs={'annotation': self.annotation, 'to_col': self.params['cluster_col']})
                # convert labels to numbers
                img = np.vectorize(cluster_names.__getitem__)(img)
                clusters1 = np.vectorize(cluster_names.__getitem__)(annotate_clustering(mpp_data.data(self.params['cluster_name']), self.annotation, self.params['cluster_name'], self.params['cluster_col']))
                # shift coords according to image padding, st coords correspond to img coords
                coords1 = (np.array([mpp_data.x, mpp_data.y]) - np.array([pad_x, pad_y])[:,np.newaxis]).astype(np.int64)
                self.log.info(f'co-occurrence for {obj_id}, with {len(mpp_data.x)} elements')
                co_occ = _co_occ_opt(coords1, coords2_list, clusters1, img, num_clusters=len(self.clusters), num_processes=num_processes)
            elif CoOccAlgo(algorithm) == CoOccAlgo.SQUIDPY:
                adata = self.mpp_data.subset(obj_ids=[obj_id], copy=True).get_adata(obs=[self.params['cluster_name']])
                # ensure that cluster annotation is present in adata
                if self.params['cluster_name'] != self.params['cluster_col']:
                    adata.obs[self.params['cluster_col']] = annotate_clustering(adata.obs[self.params['cluster_name']], self.annotation, 
                        self.params['cluster_name'], self.params['cluster_col'])
                adata.obs[self.params['cluster_col']] = adata.obs[self.params['cluster_col']].astype('category')
                self.log.info(f'co-occurrence for {obj_id}, with shape {adata.shape}')
                cur_co_occ, _ = sq.gr.co_occurrence(
                    adata,
                    cluster_key=self.params['cluster_col'],
                    spatial_key='spatial',
                    interval=interval,
                    copy=True, show_progress_bar=False,
                    n_splits=1
                )
                # ensure that co_occ has correct format incase of missing clusters
                co_occ = np.zeros((len(self.clusters),len(self.clusters),len(interval)-1))
                cur_clusters = np.vectorize(cluster_names.__getitem__)(np.array(adata.obs[self.params['cluster_col']].cat.categories))
                grid = np.meshgrid(cur_clusters, cur_clusters)
                co_occ[grid[0].flat, grid[1].flat] = cur_co_occ.reshape(-1, len(interval)-1)
            co_occs.append(co_occ.copy())
            obj_ids.append(obj_id)

            i+=1
            if (i%chunks == 0) or (obj_id == missing_obj_ids[-1]):
                # save
                self.log.info(f'Saving chunk {i-chunks}-{i}')
                # add info to adata
                co_occ = np.array(co_occs)
                for i1,c1 in enumerate(self.clusters):
                    for i2,c2 in enumerate(self.clusters):
                        df = pd.DataFrame(co_occ[:,i1,i2], index=obj_ids, columns=np.arange(len(interval)-1).astype(str))
                        df.index = df.index.astype(str)
                        # ensure obj_ids are in correct order
                        df = pd.merge(df, self.adata.obs, how='right', left_index=True, right_on='mapobject_id', suffixes=('','right'))[df.columns]
                        df = df.fillna(0)
                        # add to adata.obsm
                        if f'co_occurrence_{c1}_{c2}' in self.adata.obsm:
                            self.adata.obsm[f'co_occurrence_{c1}_{c2}'] += df
                        else:
                            self.adata.obsm[f'co_occurrence_{c1}_{c2}'] = df
                self.adata.uns['co_occurrence_params'] = {'interval': list(interval)}
                self.log.info(f'saving adata to {self.fname}')
                self.log.info(f'adata params {self.adata.uns["params"]}')
                self.adata.uns['params'] = self.params # add params to adata again, because exp_name keeps getting lost for some reason (this is a really weird bug...)
                self.adata.write(self.fname)
                # reset co_occ list and obj_ids
                co_occs = []
                obj_ids = []

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

    def _missing_co_occ_obj_ids(self):
        """
        Return those obj_ids that do not yet have co-occurence scores calculated
        """
        n = f'co_occurrence_{self.clusters[0]}_{self.clusters[0]}'
        if n not in self.adata.obsm.keys():
            # no co-occ calculated
            return self.mpp_data.unique_obj_ids
        else:
            OBJ_ID = get_data_config(self.exp.config['data']['data_config']).OBJ_ID
            masks = []
            for c1 in self.clusters:
                for c2 in self.clusters:
                    arr = self.adata.obsm[f'co_occurrence_{c1}_{c2}']
                    masks.append((self.adata.obsm[f'co_occurrence_{c1}_{c2}']==0).all(axis=1))
            obj_ids = np.array(self.adata[np.array(masks).T.all(axis=1)].obs[OBJ_ID]).astype(np.uint32)
            return obj_ids


from numba import jit
import numba.types as nt
ft = nt.float32
it = nt.int64

@jit(ft[:,:](it[:],it[:],it), fastmath=True)
def _count_co_occ(clus1:np.ndarray, clus2:np.ndarray, num_clusters:int) -> np.ndarray:
    co_occur = np.zeros((num_clusters, num_clusters), dtype=np.float32)
    for i, j in zip(clus1, clus2):
        co_occur[i, j] += 1
    return co_occur


def _co_occ_opt_helper(coords2, coords1, clusters1, img, num_clusters):
    """
    Helper function for co_occ scores. Counts occurrence of cluster pairs given lists of coords.
    """
    # NOTE: order of arguments is important here, because of call to multiprocessing.Pool.map 
    # (coords2 is iterated over and therefore needs to be first argument)
    # get img coords to consider for this interval (len(interval_coords), num obs)
    cur_coords = np.expand_dims(coords2, 2) + np.expand_dims(coords1, 1)

    # get cluster of center pixel + repeat for len(interval_coords)
    clus1 = np.tile(clusters1, [cur_coords.shape[1], 1])
    # reshape to (2, xxx)
    cur_coords = cur_coords.reshape((2,-1))
    clus1 = clus1.reshape([-1])

    # filter cur_coords that are outside image
    shape = np.expand_dims(np.array([img.shape[1],img.shape[0]]), 1)
    mask = np.all((cur_coords >= 0) & (cur_coords < shape), axis=0)
    cur_coords = cur_coords[:,mask]
    clus1 = clus1[mask]

    # get cluster of cur_coords
    clus2 = img[cur_coords[1], cur_coords[0]].flatten()

    # remove those pairs where clus2 is outside of this image (cluster id is not a valid id)
    mask = clus2 < num_clusters
    #assert (clus1 < num_clusters).all()
    clus1 = clus1[mask]
    clus2 = clus2[mask]

    co_occur = _count_co_occ(clus1, clus2, num_clusters)
    return co_occur

def _co_occ_opt(coords1: np.ndarray, # int64
               coords2_list: np.ndarray, # int64
               clusters1: np.ndarray, # int64
               img: np.ndarray, # int64
               num_clusters: int,
               num_processes = None
              ) -> np.ndarray:
    """
    Calculate co-occurrence scores for several intervals.
    
    For decreased memory usage coords1 x coords2 pairs are chunked in CO_OCC_CHUNK_SIZE chunks and processed.
    If num_processes is specified, uses multiprocessing to calculate co-occurrence scores
    
    Args:
        coords1: first list of coordiantes
        coords2_list: second list of coordinates, for different intervals
        clusters1: cluster assigments of coords1
        img: cluster image used to look up cluster assigments of coords2
        num_clusters: total number of clusters. 
            Every cluster assigment greater than this number is filtered (assuming eg background values)
        num_processes: if not none, uses multiprocessing.Pool to calculate co-occurrence scores
        
    Returns:
        co-occurrence scores in num_clusters x num_clusters x num_intervals matrix
    """
    log = logging.getLogger('_co_occ_opt')
    if num_processes is not None:
        pool = multiprocessing.Pool(num_processes)
        log.info(f'using {num_processes} processes to calculate co-occ scores.')
    if coords1.shape[1] > CO_OCC_CHUNK_SIZE:
        raise ValueError(f'coords1 with size {coords1.shape[1]} is larger than CO_OCC_CHUNK_SIZE {CO_OCC_CHUNK_SIZE}. Cannot compute _co_occ_opt')
    out = np.zeros((num_clusters, num_clusters, len(coords2_list)), dtype=np.float32)
    # iterate over each interval
    for idx, coords2 in enumerate(coords2_list):
        #log.info(f'co occ for interval {idx+1}/{len(coords2_list)}, with {coords1.shape[1]} x {coords2.shape[1]} coord pairs')
        if (coords1.shape[1] * coords2.shape[1]) > CO_OCC_CHUNK_SIZE:
            chunk_size = int(CO_OCC_CHUNK_SIZE / coords1.shape[1])
            coords2_chunks = np.split(coords2, np.arange(0,coords2.shape[1],chunk_size), axis=1)
            #log.info(f'splitting coords2 in {len(coords2_chunks)} chunks')
        else:
            coords2_chunks = [coords2]

        # calculate pairwise cluster counts
        t1 = time.time()
        co_occur = np.zeros((num_clusters, num_clusters), dtype=np.float32)
        map_fn = partial(_co_occ_opt_helper, coords1=coords1, clusters1=clusters1, img=img, num_clusters=num_clusters)
        if num_processes is not None:
            #for res in tqdm.tqdm(pool.imap_unordered(map_fn, coords2_chunks), total=len(coords2_chunks)):
            for res in pool.imap_unordered(map_fn, coords2_chunks):
                co_occur += res
        else:
            for res in map(map_fn, coords2_chunks):
                co_occur += res

        t2 = time.time()
        #log.info(f'calculating co_occur for these coords took {t2-t1:.0f}s.')

        # calculate co-occ scores
        probs_matrix = co_occur / np.sum(co_occur)
        probs = np.sum(probs_matrix, axis=1)

        probs_con = np.zeros((num_clusters, num_clusters), dtype=np.float32)
        for c in np.unique(img):
            # do not consider background value in img
            if c >= num_clusters:
                continue
            probs_conditional = co_occur[c] / np.sum(co_occur[c])
            probs_con[c, :] = probs_conditional / probs

        out[:, :, idx] = probs_con
    return out

def _prepare_co_occ(interval):
    """
    return lists of coordinates to consider for each interval. Coordinates are relative to [0,0].
    """
    arr = np.zeros((int(interval[-1])*2+1, int(interval[-1])*2+1))
    # calc distances for interval range (assuming c as center px)
    c = int(interval[-1])+1
    c = np.array([c,c]).T
    xx, yy = np.meshgrid(np.arange(len(arr)), np.arange(len(arr)))
    coords = np.array([xx.flatten(), yy.flatten()]).T
    dists = np.sqrt(np.sum((coords - c)**2, axis=-1))
    dists = dists.reshape(int(interval[-1])*2+1, -1)
    # calc coords for each thresh interval
    coord_lists = []
    for thres_min, thres_max in zip(interval[:-1], interval[1:]):
        xy = np.where((dists <= thres_max) & (dists > thres_min))
        coord_lists.append(xy - c[:,np.newaxis])

    return coord_lists