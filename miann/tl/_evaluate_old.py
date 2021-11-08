import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from pelkmans.constants import DATA_DIR, EXPERIMENT_CONFIG_FILE, EXPERIMENT_DIR
from pelkmans.utils import load_config, init_logging
from pelkmans.estimator import merged_config, Estimator
import logging
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import sklearn.cluster
import json
from tqdm import tqdm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage as fastcluster_linkage
from sklearn.metrics import silhouette_score

class Evaluator:
    """
    load estimator and data (val or test).
    contains functions for predicting images, 
    calculating clustering, and projecting clusters to images.
    Additionally contains information about training history etc.
    """
    config = {
        'img_ids': None,
        'predict_imgs': True,
        'predict_cluster_imgs': True,
        'cluster_rep': 'latent',
        'cluster_method': 'argmax', # (som_)kmeans, argmax, hierarchical_argmax, hierarchical_kmeans, (som_)leiden
        # how large should the distance between clusters be in order to be merged (for hierarchical_argmax)
        # distance between two clusters is the silhouette score of outputs (y) 
        'linkage_maxdist': 0.0,
        'kmeans_n_clusters': 10,
        'leiden_resolution': 1.0,
        'save_adata': True
    }
    def __init__(self, exp_name, experiment_dir, split='val', config={}, **kwargs):
        assert split in ('val', 'test'), "split has to be either val or test"
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Creating Evaluator for split {}'.format(split))
        self.exp_path = os.path.join(EXPERIMENT_DIR, experiment_dir, exp_name)
        self.split = split
        self.config = merged_config(self.config, config)
        
        # build estimator
        self.est = Estimator.for_evaluation(self.exp_path)
        self.input_size = self.est.model.config.get('num_neighbors', 1)
        self.num_channels = self.est.model.config.get('num_channels', 1)
        self.x = self.est.ds.data[self.split]['x']
        self.c = self.est.ds.data[self.split]['c']
        if self.est.model.is_conditional:
            self.data = self._get_samples([self.x, self.c])
        else:
            self.data = self._get_samples(self.x)
        
        # get imgs
        if kwargs.get('load_images', True):
            self.log.info('loading {} images'.format(self.split))
            (self.imgs, self.cond_imgs), self.config['img_ids'] = self.est.ds.get_imgs(self.split, self.config['img_ids'], self.est.model.is_conditional)
            # save img_ids
            np.savetxt(os.path.join(self.exp_path, 'img_ids_{}.csv'.format(self.split)), self.config['img_ids'], delimiter=',')
        
        # load history
        self.history = pd.read_csv(os.path.join(self.exp_path, 'history.csv'), index_col=0)
        self.per_channel_mse = self.calculate_mse()
        
        self.adata = None
        self.predicted_imgs = None
        self.cluster_imgs = None
        
    @classmethod
    def from_evaluated(cls, exp_name, experiment_dir, split='val', **kwargs):
        # load img_ids
        img_ids = np.loadtxt(os.path.join(EXPERIMENT_DIR, experiment_dir, exp_name, 'img_ids_{}.csv'.format(split)), delimiter=',')
        # load evaluation config
        config_fname = os.path.join(EXPERIMENT_DIR, experiment_dir, exp_name, 'config.json')
        config = json.load(open(config_fname))['evaluation']
        config['img_ids'] = img_ids.astype(int) # add img_ids in config
        self = cls(exp_name, experiment_dir, split=split, config=config, **kwargs)
        # load adata, predicted_imgs, and cluster_imgs
        self.predicted_imgs = np.load(os.path.join(self.exp_path, 'pred_imgs_epoch{:03d}_{}.npy'.format(self.est.epoch, self.split)))
        self.cluster_imgs = np.load(os.path.join(self.exp_path, 'clus_imgs_epoch{:03d}_{}.npy'.format(self.est.epoch, self.split)))
        adata_path = os.path.join(self.exp_path, 'adata_epoch{:03d}_{}.h5ad'.format(self.est.epoch, self.split))
        if os.path.exists(adata_path):
            self.adata = ad.read(adata_path)
        clus_dist_path = os.path.join(self.exp_path, 'clus_distances_epoch{:03}_{}.npz'.format(self.est.epoch, self.split))
        if os.path.exists(clus_dist_path):
            data = np.load(clus_dist_path)
            self.cluster_distances = data['distances']
            self.linkage = data['linkage']
        return self
        
    def calculate_mse(self):
        if self.est.model.is_conditional:
            pred = self.est.predict_model([self.x, self.c])
        else:
            pred = self.est.predict_model(self.x)
        true = self.est.ds.get_y(split=self.split, output_channels=self.est.output_channels)
        
        mse = np.mean((true.squeeze()-pred)**2, axis=0)
        return mse
    
    def get_mse(self, channels):
        channel_ids = self.est.ds.get_channel_ids(channels, from_channels=self.est.output_channels)
        mse = []
        for channel_id in channel_ids:
            if np.isnan(channel_id):
                mse.append(0)
            else:
                mse.append(self.per_channel_mse[int(channel_id)])
        return mse
    
    def get_predicted_imgs(self, channels):
        channel_ids = self.est.ds.get_channel_ids(channels, from_channels=self.est.output_channels)
        predicted_imgs = np.zeros(list(self.predicted_imgs.shape)[:-1] + [len(channels)], dtype=self.predicted_imgs.dtype)
        for i,channel_id in enumerate(channel_ids):
            if not np.isnan(channel_id):
                predicted_imgs[:,:,:,i] = self.predicted_imgs[:,:,:,int(channel_id)]
        return predicted_imgs
    
    def get_cluster_heatmap(self, channels=None, num_clusters=None, clustering='clustering'):
        # get clustering and num_clusters
        clustering = np.array(self.adata.obs[clustering])
        if num_clusters is None:
            clusters = np.sort(np.unique(clustering))
            num_clusters = len(clusters)
        else:
            clusters = range(0, num_clusters)
            # cast clustering to int (might be str otherwise) to fit to cluster labels
            clustering = clustering.astype('int') 
        if channels is None:
            channels = self.est.output_channels
            if channels is None: # TODO need this?
                channels = list(self.est.ds.channels.index)
        # z-score x
        x = self.est.ds.get_y(split=self.split, output_channels=channels)
        x = (x-x.mean(axis=0))/x.std(axis=0)
        arr = np.zeros((len(clusters),len(channels)))
        for i,cl in enumerate(clusters):
            for ch in range(0,len(channels)):
                arr[i,ch] = x[clustering==cl][:,ch].mean()
                if np.isnan(arr[i,ch]):
                    arr[i,ch] = 0
        arr = arr.T
        return arr, (clusters, channels)
    
    # evaluation function
    def evaluate(self, predict_imgs=None, predict_cluster_imgs=None, save_adata=None):
        self.log.info('evaluating model {} on split {}'.format(self.exp_path, self.split))
        if predict_imgs is None: 
            predict_imgs=self.config['predict_imgs']
        if predict_cluster_imgs is None: 
            predict_cluster_imgs=self.config['predict_cluster_imgs']
        if save_adata is None: 
            save_adata = self.config['save_adata']
        
        if predict_imgs and self.predicted_imgs is None:
            self.log.info('predicting {} images'.format(len(self.imgs)))
            self.predicted_imgs = self.predict_imgs(save=True)
        if (save_adata or predict_cluster_imgs) and self.adata is None:
            self.log.info('creating adata')
            self.adata = self.create_adata()
            if save_adata:
                self.adata.write(os.path.join(self.est.experiment_dir, 'adata_epoch{:03d}_{}.h5ad'.format(self.est.epoch, self.split)))
        if 'hierarchical' in self.config['cluster_method']:
            self.log.info('calculating cluster distances')
            self.cluster_distances, self.linkage = self.calc_cluster_distances(save=True)
        if predict_cluster_imgs and self.cluster_imgs is None:
            self.log.info('predicting clusters of {} images'.format(len(self.imgs)))
            self.cluster_imgs = self.predict_cluster_imgs(save=True)
    
    def create_adata(self):
        # create adata
        x = self.x[:,self.input_size//2,self.input_size//2,:]
        var = self.est.ds.channels.reset_index().rename(columns={'name':'var_names'})
        var = var.drop(labels='index', axis=1)
        obs = self.est.ds.get_metadata(self.split, columns=['mapobject_id', 'well_name', 'cell_type', 'perturbation', 'perturbation_duration', 'cell_cycle'])
        adata = ad.AnnData(x, obsm={'X_latent':self.data}, 
                           obs=obs, var=var)
        sc.pp.neighbors(adata, use_rep='X_latent')
        
        # add cluster
        if 'kmeans' in self.config['cluster_method']:
            clusters = sklearn.cluster.KMeans(n_clusters=self.config['kmeans_n_clusters']).fit_predict(self.data)
            adata.obs['clustering'] = clusters
        elif 'leiden' in self.config['cluster_method']:
            sc.tl.leiden(adata, resolution=self.config['leiden_resolution'], key_added='clustering')
        elif 'argmax' in self.config['cluster_method']:
            adata.obs['clustering'] = np.argmax(self.data, axis=-1)
        
        return adata
        
    # getting results
    def get_clustering(self, samples=None, ignore_hierarchical=True, linkage_maxdist=None):
        """return clustering of samples"""
        # create adata with clustering if not exists
        if self.adata is None:
            self.adata = self.create_adata()
        if samples is None:
            # use clustering in adata
            cluster = np.array(self.adata.obs['clustering']).astype(np.uint8)
        else:
            if self.config['cluster_method'] in ['leiden', 'som_leiden', 'som_kmeans', 
                                                 'kmeans', 'hierarchical_kmeans', 'sub_leiden']:
                # need to project clusters to new samples
                cluster = np.array(self.adata.obs['clustering']).astype(np.uint8)
                cluster = self.__class__.project_clusters(cluster, self.data, samples)
            elif self.config['cluster_method'] in ['argmax', 'hierarchical_argmax']:
                cluster = np.argmax(samples, axis=-1)
            else:
                raise NotImplementedError
        # check if need to do mapping?
        if not ignore_hierarchical and self.config['cluster_method'] in ['hierarchical_kmeans', 'hierarchical_argmax']:
            cluster_mapping = self.get_cluster_mapping(linkage_maxdist)
            cluster = self.__class__.rename_clusters(cluster, cluster_mapping)
        return cluster
                
    def calc_cluster_distances(self, save=False):
        """
        Using val/test data, calculate pairwise silhouette scores between clusters.
        Returns n_cluster x n_cluster distance matrix (0 most similar, 1 most distant)
        """
        X = self.est.ds.data[self.split]['y']
        y = self.get_clustering(ignore_hierarchical=True)
        
        # calculates silhouette scores for each pair of clusters
        num_clusters = y.max()+1
        scores = np.zeros((num_clusters, num_clusters))
        for c1 in tqdm(range(num_clusters)):
            for c2 in range(num_clusters):
                if c2 <= c1:
                    continue
                else:
                    c1_mask = y == c1
                    c2_mask = y == c2
                    if (sum(c1_mask) == 0) or (sum(c2_mask)==0):
                        print(c1, c2, 'one is zero')
                        continue
                    if (sum(c1_mask) + sum(c2_mask)) < 3:
                        print(c1, c2, 'very low numbers!')
                        continue
                    score = silhouette_score(X[c1_mask|c2_mask], y[c1_mask|c2_mask])
                    scores[c1, c2] = score
                    scores[c2, c1] = score
        # calculate linkage
        linkage = fastcluster_linkage(squareform(scores), method='ward')
        if save:
            name = 'clus_distances_epoch{:03}_{}.npz'.format(self.est.epoch, self.split)
            np.savez(os.path.join(self.est.experiment_dir, name), distances=scores, linkage=linkage)
        return scores, linkage
    
    def get_cluster_mapping(self, linkage_maxdist=None):
        if linkage_maxdist is None:
            linkage_maxdist = self.config['linkage_maxdist']
        if self.linkage is None:
            self.cluster_distances, self.linkage = self.calc_cluster_distances()  # NOTE: this will take some time
        cluster_mapping = fcluster(self.linkage, t=linkage_maxdist, criterion='distance')
        return cluster_mapping
    
    def predict_cluster_imgs(self, save=False, linkage_maxdist=None):
        input_samples = self._get_inputs_from_images(self.imgs, self.cond_imgs)
        samples = self._get_samples(input_samples)
        clusters = self.get_clustering(samples, linkage_maxdist=linkage_maxdist)
        cluster_imgs = self._reconstruct_images_from_samples(clusters, self.imgs, default_value=np.max(clusters)+1, dtype=int)
        if save:
            img_name = 'clus_imgs_epoch{:03d}_{}.npy'.format(self.est.epoch, self.split)
            save_path = os.path.join(self.est.experiment_dir, img_name)
            self.log.info('Saving cluster images to {}'.format(save_path))
            np.save(save_path, cluster_imgs)
        return cluster_imgs
        
    def predict_imgs(self, save=False):
        input_samples = self._get_inputs_from_images(self.imgs, self.cond_imgs)
        samples = self.est.predict_model(input_samples)
        pred_imgs = self._reconstruct_images_from_samples(samples, self.imgs, default_value=0, dtype=self.imgs.dtype)
        if save:
            img_name = 'pred_imgs_epoch{:03d}_{}.npy'.format(self.est.epoch, self.split)
            save_path = os.path.join(self.est.experiment_dir, img_name)
            self.log.info('Saving cluster images to {}'.format(save_path))
            np.save(save_path, pred_imgs)
        return pred_imgs
        
    # helper methods
    def _get_samples(self, x):
        """return desired representation from given inputs to the neural network"""
        if self.config['cluster_rep'] == 'latent':
            samples = self.est.model.encoder.predict(x, batch_size=128)
        elif self.config['cluster_rep'] == 'entangled':
            # this is only for cVAE models which have an "entangled" layer in the decoder
            # create the model for predicting the latent
            decoder_to_entangled_latent = tf.keras.Model(self.est.model.decoder.input, self.est.model.entangled_latent)
            encoder_to_entangled_latent = tf.keras.Model(self.est.model.input,
                                                         decoder_to_entangled_latent([self.est.model.encoder(self.est.model.input), self.est.model.input[1]]))
            samples = encoder_to_entangled_latent.predict(x, batch_size=128)
        elif self.config['cluster_rep'] == 'decoder':
            samples = self.est.predict_model(x)
        elif self.config['cluster_rep'] == 'input':
            samples = x[:,self.input_size//2,self.input_size//2,:]
        elif self.config['cluster_rep'] == 'latent_y':
            samples = self.est.model.encoder_y.predict(x, batch_size=128)
        else:
            raise NotImplementedError
        return samples
    
    # projecting to images
    def _get_inputs_from_images(self, imgs, c_imgs=None):
        input_samples = []
        input_cond = None
        if c_imgs is not None:
            input_cond = []
        for i, img in enumerate(imgs):
            mask = img[:,:,0]!=0
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                idx = tuple(slice(pp-self.input_size//2, pp+self.input_size//2+1) for pp in [y,x])
                if img[idx].shape[0] != self.input_size or img[idx].shape[1] != self.input_size:
                    input_samples.append(np.zeros((self.input_size, self.input_size, self.num_channels), dtype=np.float32))
                else:
                    input_samples.append(img[idx])
                if c_imgs is not None:
                    input_cond.append(c_imgs[i][y,x])
        input_samples = np.array(input_samples)
        if c_imgs is not None:
            input_cond = np.array(input_cond)
        if input_cond is not None:
            return [input_samples, input_cond]
        else:
            return input_samples
        
    def _reconstruct_images_from_samples(self, samples, imgs, default_value=0, dtype='float'):
        if len(samples.shape) < 2:
            samples = samples[:,np.newaxis]
        output_shape = list(imgs.shape)
        output_shape[-1] = len(samples[0])
        res_imgs = np.zeros(output_shape, dtype=dtype)+default_value
        res_imgs[imgs[:,:,:,0]!=0] = samples
        return res_imgs
    
    @staticmethod
    def project_clusters(clusters, data, samples):
        # calculate cluster representatives
        cluster_rep = []
        cluster_size = []
        cluster_cats = np.unique(clusters)
        for cat in cluster_cats:
            els = data[clusters==cat]
            cluster_rep.append(np.array(els.mean(axis=0)))
            cluster_size.append(els.shape[0])
        cluster_rep = np.array(cluster_rep)
        #print(cluster_size)
        # project cluster assignments of nearest cluster rep to samples
        projected_clusters = ((samples[:, np.newaxis] - cluster_rep[np.newaxis])**2).sum(axis=2).argmin(axis=1)
        # relabel clusters to original names
        for i, cat in enumerate(cluster_cats):
            projected_clusters[projected_clusters==i] = cat
        return projected_clusters
    
    @staticmethod
    def rename_clusters(clusters, cluster_mapping):
        res = np.zeros(len(clusters), dtype=clusters.dtype)
        for old, new in enumerate(cluster_mapping):
            res[clusters==old] = new
        return res
    
    def plot_cluster_overlap(self, compare_with=['well_name'], save=None):
        cluster_overlap, existing_mids = self.get_cluster_overlap()
        metadata = self.est.ds.metadata
        cluster_names = list(cluster_overlap.columns)

        fig, axes = plt.subplots(len(cluster_names), len(compare_with), figsize=(len(compare_with)*3,len(cluster_names)*2), squeeze=False)
        plt.suptitle(os.path.split(self.exp_path)[-1])
        for j, (y_str, col) in enumerate(zip(compare_with, axes.T)):

            col[0].set_title(y_str)
            y_names = self.adata.obs.set_index('mapobject_id')[y_str].loc[existing_mids]
            y_names = np.array(y_names[~y_names.index.duplicated(keep='first')])
            #y_names = metadata.set_index('mapobject_id')[y_str][existing_mids]
            y_labels = np.unique(y_names)
            y = np.zeros(len(y_names))
            for i,n in enumerate(y_labels):
                y[y_names==n] = i

            for cl, ax in zip(cluster_names, col):
                x = np.array(cluster_overlap[cl][existing_mids]>0)
                hist, _, _ = np.histogram2d(x, y, bins=[2,len(y_labels)], density=True)

                im = ax.imshow(hist, vmin=0, vmax=1)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['not present', 'present'])
                ax.set_xticks(range(len(y_labels)))
                ax.set_xticklabels(y_labels)
                if j == 0:
                    ax.set_ylabel('Cluster {}'.format(cl))
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        #plt.colorbar(im)
        
    def plot_cluster_heatmap(self, channels=None, num_clusters=None, ax=None, clustering='clustering'):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        heatmap, (clusters, channels) = self.get_cluster_heatmap(channels=channels, num_clusters=num_clusters, clustering=clustering)
        ax.imshow(heatmap, cmap='bwr', vmin=-5, vmax=5) 
        ax.set_yticks(range(0,len(channels)))
        ax.set_yticklabels(np.array(channels))
        ax.set_xticks(range(0,len(clusters)))
        ax.set_xticklabels(np.array(clusters))
    
    def get_cluster_overlap(self, cluster='clustering'):
        clusters = np.array(self.adata.obs[cluster])
        mapobject_id = np.array(self.adata.obs.mapobject_id)
        metadata = self.est.ds.metadata

        cluster_occurrence = pd.DataFrame({'mapobject_id': metadata.mapobject_id,})
        cluster_occurrence = cluster_occurrence.set_index('mapobject_id')
        for c in np.unique(clusters):
            ids = mapobject_id[clusters==c]
            unique_ids, counts = np.unique(ids, return_counts=True)
            cluster_occurrence[c] = pd.Series(counts, index=unique_ids)

        existing_mids = np.unique(mapobject_id)
        return cluster_occurrence, existing_mids


class ModelComparator:
    def __init__(self, evas_dict, save_dir=None):
        self.evas = evas_dict
        self.exp_names = list(self.evas.keys())
        self.save_dir = save_dir
        # TODO maybe some assertations that output_channels are the same and that 
    
    @classmethod
    def from_exp_names(cls, exp_names, experiment_dir):
        evas = {exp_name:Evaluator.from_evaluated(exp_name, experiment_dir) for exp_name in exp_names}
        return cls(evas)
    
    def plot_history(self, values=['loss'], exp_names=None, save_prefix=''):
        """line plot of values against epochs for different experiments
        Args:
            values: key in history dict to be plotted
            exp_names (optional): compare only a subset of experiments
        """
        if exp_names is None:
            exp_names = self.exp_names
        cmap = plt.get_cmap('tab10')
        cnorm  = colors.Normalize(vmin=0, vmax=10)
        fig, axes = plt.subplots(1,len(values), figsize=(len(values)*5,5), sharey=True)
        if len(values) == 1:
            # make axes iterable, even if there is only one ax
            axes = [axes]
        for val, ax in zip(values, axes):
            ax.set_title(val)
            for i,exp_name in enumerate(exp_names):
                color = cm.viridis(i)
                hist = self.evas[exp_name].history
                if val in hist.keys():
                    ax.plot(hist.index, hist[val], label=exp_name, color=cmap(cnorm(i)))
            ax.legend()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}history.png'.format(save_prefix)), dpi=100)
            
    def plot_final_score(self, score='loss', fallback_score='loss', exp_names=None, save_prefix=''):
        if exp_names is None:
            exp_names = self.exp_names
        scores = []
        for exp_name in exp_names:
            scores.append(list(self.evas[exp_name].history.get(score, self.evas[exp_name].history[fallback_score]))[-1])
        fig, ax = plt.subplots(1,1)
        ax.bar(x=range(len(scores)), height=scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(exp_names, rotation=45)
        ax.set_title(score + '('+fallback_score+')')
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}final.png'.format(save_prefix)), dpi=100)
            
    def plot_per_channel_mse(self, exp_names=None, channels=None, save_prefix=''):
        if exp_names is None:
            exp_names = self.exp_names
        if channels is None:
            channels = self.evas[exp_names[0]].est.output_channels
            if channels is None:
                channels = list(self.evas[exp_names[0]].est.ds.channels.index)
        offset = 0.9 / len(exp_names)
        fig, ax = plt.subplots(1,1, figsize=(20,5))
        X = np.arange(len(channels))
        for i,exp_name in enumerate(exp_names):
            ax.bar(X+offset*i, self.evas[exp_name].get_mse(channels), label=exp_name, width=offset)
        ax.set_xticks(X+0.5)
        ax.set_xticklabels(channels, rotation=90)
        ax.legend()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}per_channel_mse.png'.format(save_prefix)), dpi=100)
            
    def plot_predicted_images(self, exp_names=None, channels=None, img_ids=None, save_prefix=''):
        if exp_names is None:
            exp_names = self.exp_names
        if channels is None:
            channels = self.evas[exp_names[0]].est.output_channels
            if channels is None:
                channels = list(self.evas[exp_names[0]].est.ds.channels.index)
        
        # get input images
        input_imgs = self.evas[exp_names[0]].imgs
        if img_ids is None:
            img_ids = np.arange(len(input_imgs))
        # restrict to channels
        input_channel_ids = self.evas[exp_names[0]].est.ds.get_channel_ids(channels)
        input_imgs = input_imgs[:,:,:,input_channel_ids]
        
        # get predicted images
        predicted_imgs = []
        for exp_name in exp_names:
            pred_imgs = self.evas[exp_name].get_predicted_imgs(channels)
            channel_ids = self.evas[exp_name].est.ds.get_channel_ids(channels, from_channels=self.evas[exp_name].est.output_channels)
            predicted_imgs.append(pred_imgs)
        
        # plot
        for img_id in img_ids:
            fig, axes = plt.subplots(len(exp_names)+1,len(channels), figsize=(len(channels)*2,2*(len(exp_names)+1)))
            for i, col in enumerate(axes.T):
                col[0].imshow(input_imgs[img_id,:,:,i], vmin=0, vmax=1)
                if i == 0:
                    col[0].set_ylabel('Groundtruth')
                col[0].set_title(channels[i])
                for j, ax in enumerate(col[1:]):
                    ax.imshow(predicted_imgs[j][img_id,:,:,i], vmin=0, vmax=1)
                    if i == 0:
                        ax.set_ylabel(exp_names[j])
            for ax in axes.flat:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            if self.save_dir is not None:
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_dir, '{}predicted_images_{}.png'.format(save_prefix, img_id)), dpi=300)

    def plot_cluster_images(self, exp_names=None, img_ids=None, img_labels=None, img_channel='00_DAPI', save_prefix='', cmap='gist_ncar'):
        if exp_names is None:
            exp_names = self.exp_names
        # get input images
        input_imgs = self.evas[exp_names[0]].imgs
        if img_ids is None:
            img_ids = np.arange(len(input_imgs))
        # restrict to channel
        channel_id = self.evas[exp_names[0]].est.ds.get_channel_ids([img_channel])[0]
        if not isinstance(channel_id, int):
            print(f'WARNING: plot_cluster_images: desired channel {img_channel} does not exist, using channel 0 instead')
            channel_id = 0
        input_imgs = input_imgs[:,:,:,channel_id]
        
        # get cluster images
        cluster_imgs = []
        for exp_name in exp_names:
            cluster_imgs.append(self.evas[exp_name].cluster_imgs)
        # calculate num clusters for consistent color maps
        num_clusters = [len(np.unique(clus_imgs)) for clus_imgs in cluster_imgs]
        
        # plot results
        fig, axes = plt.subplots(len(img_ids),len(cluster_imgs)+1, figsize=(3*(len(cluster_imgs)+1),len(img_ids)*3))
        for cid, row in zip(img_ids,axes):
            row[0].imshow(input_imgs[cid], vmin=0, vmax=1)
            ylabel = 'Object id {} '.format(cid)
            if img_labels is not None:
                ylabel = ylabel + img_labels[cid]
            row[0].set_ylabel(ylabel)
            if cid == 0:
                row[0].set_title('channel {}'.format(img_channel))
            for i, ax in enumerate(row[1:]):
                ax.imshow(cluster_imgs[i][cid,:,:,0], cmap=cmap, vmin=0, vmax=num_clusters[i])
                if cid == 0:
                    ax.set_title(exp_names[i])
        for ax in axes.flat:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.tight_layout()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}cluster_images.png'.format(save_prefix)), dpi=100)

    def plot_cluster_heatmap(self, exp_names=None, channels=None, num_clusters=None, save_prefix=''):
        if exp_names is None:
            exp_names = self.exp_names
        nrows = int(np.ceil(len(exp_names)/3.))
        fig, axes = plt.subplots(nrows,3, figsize=(20,8*nrows))
        for i,exp_name in enumerate(exp_names):
            ax = axes.flat[i]
            self.evas[exp_name].plot_cluster_heatmap(channels=channels, num_clusters=num_clusters, ax=ax)
            ax.set_title(exp_name)
        for ax in axes.flat[len(exp_names):]:
            ax.axis('off')
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}heatmap.png'.format(save_prefix)))
            
    def plot_cluster_overlap(self, exp_names=None, compare_with=['well_name']):
        if exp_names is None:
            exp_names = self.exp_names
            
        for exp_name in exp_names:
            save=None
            if self.save_dir is not None:
                save = os.path.join(self.save_dir, 'cluster_overlap_{}.png'.format(exp_name))
            self.evas[exp_name].plot_cluster_overlap(compare_with=compare_with, save=save)
            
    def plot_annotated_cluster_overview(self, exp_names=None, cluster_labels={}, img_id=0, save_prefix=''):
        if exp_names is None:
            exp_names = self.exp_names
        # use marker channels
        channels = ['15_SON', '09_SRRM2', '18_NONO', '21_COIL', '16_H3', '07_H2B', '11_PML', '20_SP100', '21_NCL', '00_DAPI']
        channel_labels = ['nuclear speckle', 'nuclear speckle', 'paraspeckle', 'cajal bodies', 'histone', 'histone', 'pml', 'pml', 'nucleolus', 'dna']
        channel_ids = self.evas[exp_names[0]].est.ds.get_channel_ids(channels)

        cmaps = ['Reds', 'Greens', 'Blues', 'Greys']
        labels = np.unique(channel_labels)

        fig, axes = plt.subplots(len(exp_names)+1, len(labels), figsize=(20,(len(exp_names)+1)*3))
        # first plot channels
        for lbl, ax in zip(labels, axes[0].flat):
            ax.set_title(lbl)
            channel_idxs = np.where(np.array(channel_labels) == lbl)[0]
            for i,idx in enumerate(channel_idxs):
                ax.imshow(self.evas[exp_names[0]].imgs[img_id][:,:,channel_ids[idx]], alpha=0.5, cmap=cmaps[i])
        # then plot clusters for each experiment
        for j, exp_name in enumerate(exp_names):
            axes[j+1,0].set_ylabel(exp_name)
            for lbl, ax in zip(labels, axes[j+1].flat):
                cluster_idxs = np.where(np.array(cluster_labels[exp_name]) == lbl)[0]
                for i,idx in enumerate(cluster_idxs):
                    ax.imshow(self.evas[exp_name].cluster_imgs[img_id,:,:,0]==idx, alpha=0.5, cmap=cmaps[i], vmin=0, vmax=1)

        for ax in axes.flat:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}cluster_overview_{}.png'.format(save_prefix, img_id)), dpi=300)
    
    
