# TODO working on evaluation + clustering pipeline for exp model. 
# TODO when finished: create fns for how to aggregate - should easily be possible with this code!
import numpy as np
import os
import tensorflow as tf
import logging
from miann.tl import Estimator, Experiment
from miann.pl._plot import annotate_img
import matplotlib.pyplot as plt
from matplotlib import cm, colors

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

    # TODO might not need?
    def calculate_mse(self, mpp_data):
        """
        Calculate mean squared error from mpp_data. If mpp_data does not have decoder representation, predict it
        """
        if mpp_data.data('decoder') is None:
            self.predict(mpp_data, reps=['decoder'])
        return np.mean((mpp_data.center_mpp - mpp_data.data('decoder'))**2, axis=0)

    def predict(self, mpp_data, save_dir=None, reps=['latent'], mpp_params={}):
        """
        Predict reps from mpp_data, 
        
        Args:
        save_dir: save predicted reps to this dir (absolute path)
        reps: which representations to predict
        mpp_params: base_data_dir and subset information to save alongside of predicted mpp_data. See MPPData.write()

        Returns:
            MPPData with keys reps.
        """
        self.log.info(f'Predicting representation {reps} for mpp_data')
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


class ModelComparator:
    def __init__(self, exps, save_dir=None):
        """
        Compare experiments 
        """
        self.exps = {exp.name: exp for exp in exps}
        self.exp_names = list(self.exps.keys())
        self.save_dir = save_dir

        # default channels and mpp data
        self.mpps = {exp_name: self.exps[exp_name].get_split_mpp_data() for exp_name in self.exp_names}
        self.img_mpps = {exp_name: self.exps[exp_name].get_split_imgs_mpp_data() for exp_name in self.exp_names}
        channels = {exp_name: self.exps[exp_name].config['data'].get('output_channels', None) for exp_name in self.exp_names}
        self.channels = {key: val if val is not None else self.mpps[key].channels['name'] for key, val in channels.items()}
        
    @classmethod
    def from_dir(cls, exp_names, exp_dir):
        """
        Initialise from experiments in experiment dir
        """
        exps = [Experiment.from_dir(os.path.join(exp_dir, exp_name)) for exp_name in exp_names]
        return cls(exps, save_dir=exp_dir)

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
                hist = self.exps[exp_name].get_history()
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
            hist = self.exps[exp_name].get_history()
            scores.append(list(hist.get(score, hist[fallback_score]))[-1])
        fig, ax = plt.subplots(1,1)
        ax.bar(x=range(len(scores)), height=scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(exp_names, rotation=45)
        ax.set_title(score + '('+fallback_score+')')
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}final.png'.format(save_prefix)), dpi=100)

    def plot_per_channel_mse(self, exp_names=None, channels=None, save_prefix=''):
        def mse(mpp_data):
            return np.mean((mpp_data.center_mpp - mpp_data.data('decoder'))**2, axis=0)
        # setup: get experiments + mpps
        if exp_names is None:
            exp_names = self.exp_names
        if channels is None:
            channels = self.channels[exp_names[0]]

        # calculate mse
        mse_scores = {exp_name: mse(self.mpps[exp_name]) for exp_name in exp_names}
        channel_ids = {exp_name: self.mpps[exp_name].get_channel_ids(channels) for exp_name in exp_names}

        # plot mse
        offset = 0.9 / len(exp_names)
        fig, ax = plt.subplots(1,1, figsize=(20,5))
        X = np.arange(len(channels))
        for i,exp_name in enumerate(exp_names):
            ax.bar(X+offset*i, mse_scores[exp_name][channel_ids[exp_name]], label=exp_name, width=offset)
        ax.set_xticks(X+0.5)
        ax.set_xticklabels(channels, rotation=90)
        ax.legend()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, '{}per_channel_mse.png'.format(save_prefix)), dpi=100)

    def plot_predicted_images(self, exp_names=None, channels=None, img_ids=None, save_prefix='', **kwargs):
        """
        kwargs passed to mpp_data.get_object_imgs
        """
        if exp_names is None:
            exp_names = self.exp_names
        if channels is None:
            channels = self.channels[exp_names[0]]
        output_channels = self.exps[exp_names[0]].config['data'].get('output_channels', None)
        output_channel_ids = {exp_name: self.img_mpps[exp_name].get_channel_ids(channels, from_channels=output_channels) for exp_name in exp_names}

        # get input images
        input_channel_ids = self.img_mpps[exp_names[0]].get_channel_ids(channels)
        input_imgs = self.img_mpps[exp_names[0]].get_object_imgs(channel_ids=input_channel_ids, **kwargs)
        if img_ids is None:
            img_ids = range(len(input_imgs))
        
        # get predicted images
        predicted_imgs = []
        for exp_name in exp_names:
            pred_imgs = self.img_mpps[exp_name].get_object_imgs(data='decoder', channel_ids=output_channel_ids[exp_name], **kwargs)
            predicted_imgs.append(pred_imgs)
        
        # plot
        for img_id in img_ids:
            fig, axes = plt.subplots(len(exp_names)+1,len(channels), figsize=(len(channels)*2,2*(len(exp_names)+1)), squeeze=False)
            for i, col in enumerate(axes.T):
                col[0].imshow(input_imgs[img_id][:,:,i], vmin=0, vmax=1)
                if i == 0:
                    col[0].set_ylabel('Groundtruth')
                col[0].set_title(channels[i])
                for j, ax in enumerate(col[1:]):
                    ax.imshow(predicted_imgs[j][img_id][:,:,i], vmin=0, vmax=1)
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


    def plot_cluster_images(self, exp_names=None, img_ids=None, img_labels=None, img_channel='00_DAPI', save_prefix='', 
        rep='clustering', **kwargs):
        if exp_names is None:
            exp_names = self.exp_names

        # get input images
        input_channel_ids = self.img_mpps[exp_names[0]].get_channel_ids([img_channel])
        input_imgs = self.img_mpps[exp_names[0]].get_object_imgs(channel_ids=input_channel_ids, **kwargs)
        if img_ids is None:
            img_ids = np.arange(len(input_imgs))
        
        # get cluster images
        cluster_imgs = []
        for exp_name in exp_names:
            # load annotation
            cluster_annotation = self.exps[exp_name].get_split_cluster_annotation(rep)
            cluster_img = self.img_mpps[exp_name].get_object_imgs(data=rep, annotation_kwargs={'color': True, 'annotation':cluster_annotation}, **kwargs)
            cluster_imgs.append(cluster_img)
        # calculate num clusters for consistent color maps
        num_clusters = [len(np.unique(clus_imgs)) for clus_imgs in cluster_imgs]
        
        # plot results
        fig, axes = plt.subplots(len(img_ids),len(cluster_imgs)+1, figsize=(3*(len(cluster_imgs)+1),len(img_ids)*3), squeeze=False)
        for cid, row in zip(img_ids,axes):
            row[0].imshow(input_imgs[cid][:,:,0], vmin=0, vmax=1)
            ylabel = 'Object id {} '.format(cid)
            if img_labels is not None:
                ylabel = ylabel + img_labels[cid]
            row[0].set_ylabel(ylabel)
            if cid == 0:
                row[0].set_title('channel {}'.format(img_channel))
            for i, ax in enumerate(row[1:]):
                ax.imshow(cluster_imgs[i][cid], vmin=0, vmax=num_clusters[i])
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

