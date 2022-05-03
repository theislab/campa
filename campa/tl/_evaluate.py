from typing import Any, Dict, List, Union, Mapping, Iterable, Optional
import os
import logging

from matplotlib import cm, colors
import numpy as np
import scanpy as sc
import tensorflow as tf
import matplotlib.pyplot as plt

from campa.data import MPPData
from campa.tl._estimator import Estimator
from campa.tl._experiment import Experiment
from campa.data._conditions import process_condition_desc


class Predictor:
    """
    Predict results from trained model.

    Parameters
    ----------
    exp
        Trained Experiment.
    batch_size
        Batch size to use for prediction.
        If None, training batch size is used.
    """

    def __init__(self, exp: Experiment, batch_size: Optional[int] = None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp.set_to_evaluate()
        self.log.info(f"Creating Predictor for {self.exp.dir}/{self.exp.name}")
        # set batch size
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = self.exp.config["evaluation"].get("batch_size", self.exp.config["training"]["batch_size"])

        # build estimator
        self.est = Estimator(exp)

    def evaluate_model(self):
        """
        Predict val/test split and images.

        Uses :meth:`Experiment.evaluate_config` for settings, and calls :meth:`Predictor.predict_split`.
        """
        config = self.exp.evaluate_config
        # predict split
        self.predict_split(config["split"], reps=config["predict_reps"])
        if config["predict_imgs"]:
            self.predict_split(
                config["split"] + "_imgs",
                img_ids=config["img_ids"],
                reps=config["predict_reps"],
            )

    # TODO might not need?
    # def calculate_mse(self, mpp_data):
    #    """
    #    Calculate mean squared error from mpp_data. If mpp_data does not have decoder representation, predict it
    #    """
    #    if mpp_data.data("decoder") is None:
    #        self.predict(mpp_data, reps=["decoder"])
    #    return np.mean((mpp_data.center_mpp - mpp_data.data("decoder")) ** 2, axis=0)

    def predict(
        self,
        mpp_data: MPPData,
        save_dir: Optional[str] = None,
        reps: Iterable[str] = ("latent",),
        mpp_params: Optional[Mapping[str, Any]] = None,
    ) -> MPPData:
        """
        Predict representations from ``mpp_data``.

        Parameters
        ----------
        mpp_data
            Data to predict.
        save_dir
            Save predicted representations to this dir (absolute path).
        reps
            Which representations to predict. See :meth:`Predictor.get_representation`.
        mpp_params
            Base data dir and subset information to save alongside of predicted MPPData.
            See :meth:`MPPData.write`.

        Returns
        -------
        :class:`MPPData`
            Data with representations stored in :meth:`MPPData.data`.
        """
        self.log.info(f"Predicting representation {reps} for mpp_data")
        for rep in reps:
            mpp_data._data[rep] = self.get_representation(mpp_data, rep=rep)
        if save_dir is not None:
            mpp_data.write(save_dir, save_keys=reps, mpp_params=mpp_params)
        return mpp_data

    def predict_split(
        self,
        split: str,
        img_ids: Optional[Union[np.ndarray, List[int], int]] = None,
        reps: Iterable[str] = ("latent", "decoder"),
        **kwargs: Any,
    ) -> None:
        """
        Predict data from train/val/test split of dataset that the model was trained with.

        Saves results in ``experiment_dir/exp_name/split``.

        Parameters
        ----------
        split
            Data split to predict. One of `train`, `val`, `test`, `val_imgs`, `test_imgs`.
        img_ids
            obj_ids or number of objects that should be predicted (only for val_imgs and test_imgs).
        reps
            Representations to predict. See :meth:`Predictor.get_representation`.
        """
        self.log.info(f"Predicting split {split} for {self.exp.dir}/{self.exp.name}")
        if "_imgs" in split:
            mpp_data = self.est.ds.imgs[split.replace("_imgs", "")]
            if img_ids is None:
                img_ids = list(mpp_data.unique_obj_ids)
            if isinstance(img_ids, int):
                # choose random img_ids from available ones
                # TODO this uses new rng, before used old default np.random.choice. Will choose different cells
                rng = np.random.default_rng(seed=42)
                img_ids = rng.choice(mpp_data.unique_obj_ids, img_ids, replace=False)
            # subset mpp_data to these img_ids
            mpp_data.subset(obj_ids=img_ids)  # type: ignore[arg-type]
            # add neighborhood to mpp_data (other processing is already done)
            if self.est.ds.params["neighborhood"]:
                mpp_data.add_neighborhood(size=self.est.ds.params["neighborhood_size"])
        else:
            mpp_data = self.est.ds.data[split]

        for rep in reps:
            mpp_data._data[rep] = self.get_representation(mpp_data, rep=rep)
        save_dir = os.path.join(self.exp.full_path, f"results_epoch{self.est.epoch:03d}", split)
        # base data dir for correct recreation of mpp_data
        base_data_dir = os.path.join("datasets", self.est.ds.params["dataset_name"], split)
        mpp_data.write(
            save_dir,
            save_keys=reps,
            mpp_params={"base_data_dir": base_data_dir, "subset": True},
        )

    def get_representation(self, mpp_data: MPPData, rep: str = "latent") -> Any:
        """
        Return representation from given mpp_data inputs.

        Representation `input` returns ``mpp_data.mpp``.
        For representations `latent` and `decoder`, predict the model.

        Parameters
        ----------
        mpp_data
            Data to get representation from.
        rep
            Representation, one of: `input`, `latent`, `decoder`.

        Returns
        -------
        Iterable
            Representation.
        """
        #  TODO might remove entangled, latent_y in the future (not needed currently)
        if rep == "input":
            # return mpp
            return mpp_data.center_mpp
        # need to prepare input to model
        if self.est.model.is_conditional:
            data: List[np.ndarray] = [mpp_data.mpp, mpp_data.conditions]  # type: ignore[list-item]
        else:
            data: np.ndarray = mpp_data.mpp  # type: ignore[no-redef]
        # get representations
        if rep == "latent":
            return self.est.model.encoder.predict(data, batch_size=self.batch_size)
        elif rep == "entangled":
            # this is only for cVAE models which have an "entangled" layer in the decoder
            # create the model for predicting the latent
            decoder_to_entangled_latent = tf.keras.Model(self.est.model.decoder.input, self.est.model.entangled_latent)
            encoder_to_entangled_latent = tf.keras.Model(
                self.est.model.input,
                decoder_to_entangled_latent(
                    [
                        self.est.model.encoder(self.est.model.input),
                        self.est.model.input[1],
                    ]
                ),
            )
            return encoder_to_entangled_latent.predict(data, batch_size=self.batch_size)
        elif rep == "decoder":
            return self.est.predict_model(data, batch_size=self.batch_size)
        elif rep == "latent_y":
            return self.est.model.encoder_y.predict(data, batch_size=self.batch_size)
        else:
            raise NotImplementedError(rep)


class ModelComparator:
    """
    Compare experiments.

    Creates and saves comparison plots.

    Parameters
    ----------
    exps
        Experiments to compare.
    save_dir
        Absolute path to directory in which the plots should be saved.
    """

    def __init__(self, exps: Iterable[Experiment], save_dir: Optional[str] = None):

        self.exps = {exp.name: exp for exp in exps}
        self.exp_names: List[str] = list(self.exps.keys())
        self.save_dir: Optional[str] = save_dir
        """
        Directory where plots are saved. If None, no plots are saved.
        """

        # default channels and mpp data
        self.mpps: Dict[str, MPPData] = {}
        self.img_mpps: Dict[str, MPPData] = {}
        for exp_name in self.exp_names:
            mpp = self.exps[exp_name].get_split_mpp_data()
            assert mpp is not None
            self.mpps[exp_name] = mpp
            img_mpp = self.exps[exp_name].get_split_imgs_mpp_data()
            assert img_mpp is not None
            self.img_mpps[exp_name] = img_mpp
        channels = {
            exp_name: self.exps[exp_name].config["data"].get("output_channels", None) for exp_name in self.exp_names
        }
        self.channels: Dict[str, List[str]] = {
            key: val if val is not None else self.mpps[key].channels["name"] for key, val in channels.items()
        }

    @classmethod
    def from_dir(cls, exp_names: Iterable[str], exp_dir: str) -> "ModelComparator":
        """
        Initialise from experiments in experiment dir.

        `save_dir` will be set to `exp_dir`.

        Parameters
        ----------
        exp_names
            Name of experiments to compare.
        exp_dir
            Experiment dir in which experiments are stored
        """
        exps = [Experiment.from_dir(os.path.join(exp_dir, exp_name)) for exp_name in exp_names]
        return cls(exps, save_dir=exp_dir)

    def _filter_trainable_exps(self, exp_names: Iterable[str]) -> List[str]:
        """
        Return exp_names that are trainable.

        Needed for some plotting fns that only make sense for trainable exps.
        """
        return [exp_name for exp_name in exp_names if self.exps[exp_name].is_trainable]

    def plot_history(
        self, values: Iterable[str] = ("loss",), exp_names: Optional[Iterable[str]] = None, save_prefix: str = ""
    ) -> None:
        """
        Line plot of values against epochs for different experiments.

        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        values
            Key in history dict to be plotted.
        exp_names
            Compare only a subset of experiments.
        save_prefix
            Plot is saved under `{save_prefix}umap_{exp_name}.png`.

        Returns
        -------
        Nothing, plots and saves lineplots
        """
        if exp_names is None:
            exp_names = self.exp_names
        exp_names = self._filter_trainable_exps(exp_names)

        cmap = plt.get_cmap("tab10")
        cnorm = colors.Normalize(vmin=0, vmax=10)
        fig, axes = plt.subplots(1, len(list(values)), figsize=(len(list(values)) * 5, 5), sharey=True)
        if len(list(values)) == 1:
            # make axes iterable, even if there is only one ax
            axes = [axes]
        for val, ax in zip(values, axes):
            ax.set_title(val)
            for i, exp_name in enumerate(exp_names):
                cm.viridis(i)
                hist = self.exps[exp_name].get_history()
                assert hist is not None, "no history available"
                if val in hist.keys():
                    ax.plot(hist.index, hist[val], label=exp_name, color=cmap(cnorm(i)))
            ax.legend()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.save_dir, f"{save_prefix}history.png"),
                dpi=100,
            )

    def plot_final_score(
        self,
        score: str = "loss",
        fallback_score: str = "loss",
        exp_names: Optional[Iterable[str]] = None,
        save_prefix: str = "",
    ) -> None:
        """
        Bar plot of scores for different experiments.

        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        score
            Key in history dict to be plotted.
        fallback_score
            If ``score`` does not exist for an experiment, use ``fallback_score``.
        exp_names
            Compare only a subset of experiments.
        save_prefix
            Plot is saved under `{save_prefix}final.png`.

        Returns
        -------
        Nothing, plots and saves barplot.
        """
        if exp_names is None:
            exp_names = self.exp_names
        exp_names = self._filter_trainable_exps(exp_names)

        scores = []
        for exp_name in exp_names:
            hist = self.exps[exp_name].get_history()
            assert hist is not None, "no history available"
            scores.append(list(hist.get(score, hist[fallback_score]))[-1])
        fig, ax = plt.subplots(1, 1)
        ax.bar(x=range(len(scores)), height=scores)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(exp_names, rotation=45)
        ax.set_title(score + "(" + fallback_score + ")")
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"{save_prefix}final.png"), dpi=100)

    def plot_per_channel_mse(
        self, exp_names: Optional[List[str]] = None, channels: Optional[List[str]] = None, save_prefix: str = ""
    ) -> None:
        """
        Bar plot of MSE score per each channel for different experiments.

        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        exp_names
            Compare only a subset of experiments.
        channels
            Channels that should be visualised. If None, all channels are plotted.
        save_prefix
            Plot is saved under `{save_prefix}per_channel_mse.png`.

        Returns
        -------
        Nothing, plots and saves barplot.
        """

        def mse(mpp_data):
            return np.mean((mpp_data.center_mpp - mpp_data.data("decoder")) ** 2, axis=0)

        # setup: get experiments + mpps
        if exp_names is None:
            exp_names = self.exp_names
        exp_names = self._filter_trainable_exps(exp_names)

        if channels is None:
            channels = self.channels[exp_names[0]]

        # calculate mse
        mse_scores = {exp_name: mse(self.mpps[exp_name]) for exp_name in exp_names}
        channel_ids = {exp_name: self.mpps[exp_name].get_channel_ids(channels) for exp_name in exp_names}

        # plot mse
        offset = 0.9 / len(exp_names)
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        X = np.arange(len(channels))
        for i, exp_name in enumerate(exp_names):
            ax.bar(
                X + offset * i,
                mse_scores[exp_name][channel_ids[exp_name]],
                label=exp_name,
                width=offset,
            )
        ax.set_xticks(X + 0.5)
        ax.set_xticklabels(channels, rotation=90)
        ax.legend()
        if self.save_dir is not None:
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.save_dir, f"{save_prefix}per_channel_mse.png"),
                dpi=100,
            )

    def plot_predicted_images(
        self,
        exp_names: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        img_ids: Optional[Iterable[int]] = None,
        save_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Plot reconstructed cell images.

        Visualises a ``#exp_names x #channels`` grid for each ``img_id``.
        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        exp_names
            Compare only a subset of experiments.
        channels
            Channels that should be visualised. If None, all channels are plotted.
        img_ids
            Cell images from :attr:`ModelComparator.img_mpps` that should be visualised.
        save_prefix
            Plot is saved under `{save_prefix}predicted_images_{img_id}.png`.
        kwargs
            Passed to :meth:`MPPData.get_object_imgs`.

        Returns
        -------
        Nothing, plots and saves images.
        """
        if exp_names is None:
            exp_names = self.exp_names
        exp_names = self._filter_trainable_exps(exp_names)

        if channels is None:
            channels = self.channels[exp_names[0]]
        output_channels = self.exps[exp_names[0]].config["data"].get("output_channels", None)
        output_channel_ids = {
            exp_name: self.img_mpps[exp_name].get_channel_ids(channels, from_channels=output_channels)
            for exp_name in exp_names
        }

        # get input images
        input_channel_ids = self.img_mpps[exp_names[0]].get_channel_ids(channels)
        input_imgs = self.img_mpps[exp_names[0]].get_object_imgs(channel_ids=input_channel_ids, **kwargs)
        if img_ids is None:
            img_ids = range(len(input_imgs))

        # get predicted images
        predicted_imgs = []
        for exp_name in exp_names:
            pred_imgs = self.img_mpps[exp_name].get_object_imgs(
                data="decoder", channel_ids=output_channel_ids[exp_name], **kwargs
            )
            predicted_imgs.append(pred_imgs)

        # plot
        for img_id in img_ids:
            fig, axes = plt.subplots(
                len(exp_names) + 1,
                len(channels),
                figsize=(len(channels) * 2, 2 * (len(exp_names) + 1)),
                squeeze=False,
            )
            for i, col in enumerate(axes.T):
                col[0].imshow(input_imgs[img_id][:, :, i], vmin=0, vmax=1)
                if i == 0:
                    col[0].set_ylabel("Groundtruth")
                col[0].set_title(channels[i])
                for j, ax in enumerate(col[1:]):
                    ax.imshow(predicted_imgs[j][img_id][:, :, i], vmin=0, vmax=1)
                    if i == 0:
                        ax.set_ylabel(exp_names[j])
            for ax in axes.flat:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            if self.save_dir is not None:
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.save_dir,
                        f"{save_prefix}predicted_images_{img_id}.png",
                    ),
                    dpi=300,
                )

    def plot_cluster_images(
        self,
        exp_names: Optional[List[str]] = None,
        img_ids: Optional[List[int]] = None,
        img_labels: Optional[List[str]] = None,
        img_channel: Optional[str] = None,
        save_prefix: str = "",
        rep: str = "clustering",
        **kwargs: Any,
    ) -> None:
        """
        Plot clustering on cell images for all experiments.

        Visualises a ``#exp_names x #img_ids`` grid.
        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        exp_names
            Compare only a subset of experiments.
        img_ids
            Cell images from :attr:`ModelComparator.img_mpps` that should be visualised.
        img_labels
            Additional information to label plotted images by.
            If defined, must be a list of same length as ``img_ids``.
        img_channel
            First column of the plot is an intensity image of ``img_channel``.
            Default is the first channel defined in :attr:`MPPData.channels`.
        save_prefix
            Plot is saved under `{save_prefix}predicted_images_{img_id}.png`.
        rep
            Representation to plot. Must be a key in :meth:`MPPData.data`.
        kwargs
            Passed to :meth:`MPPData.get_object_imgs`.

        Returns
        -------
        Nothing, plots and saves images.
        """
        if exp_names is None:
            exp_names = self.exp_names

        # get input images
        if img_channel is None:
            input_channel_ids = [0]
            img_channel = str(self.img_mpps[exp_names[0]].channels.loc[0])
        input_channel_ids = self.img_mpps[exp_names[0]].get_channel_ids([img_channel])
        input_imgs = self.img_mpps[exp_names[0]].get_object_imgs(channel_ids=input_channel_ids, **kwargs)
        if img_ids is None:
            img_ids = np.arange(len(input_imgs))

        # get cluster images
        cluster_imgs = []
        for exp_name in exp_names:
            # load annotation
            cluster_annotation = self.exps[exp_name].get_split_cluster_annotation(rep)
            cluster_img = self.img_mpps[exp_name].get_object_imgs(
                data=rep,
                annotation_kwargs={"color": True, "annotation": cluster_annotation},
                **kwargs,
            )
            cluster_imgs.append(cluster_img)
        # calculate num clusters for consistent color maps
        num_clusters = [len(np.unique(clus_imgs)) for clus_imgs in cluster_imgs]

        # plot results
        fig, axes = plt.subplots(
            len(img_ids),
            len(cluster_imgs) + 1,
            figsize=(3 * (len(cluster_imgs) + 1), len(img_ids) * 3),
            squeeze=False,
        )
        for cid, row in zip(img_ids, axes):
            row[0].imshow(input_imgs[cid][:, :, 0], vmin=0, vmax=1)
            ylabel = f"Object id {cid} "
            if img_labels is not None:
                ylabel = ylabel + img_labels[cid]
            row[0].set_ylabel(ylabel)
            if cid == 0:
                row[0].set_title(f"channel {img_channel}")
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
            plt.savefig(
                os.path.join(self.save_dir, f"{save_prefix}cluster_images.png"),
                dpi=100,
            )

    def plot_umap(
        self, exp_names: Optional[Iterable[str]] = None, channels: Optional[List[str]] = None, save_prefix: str = ""
    ) -> None:
        """
        Plot UMAP representation for every experiment.

        Plots conditions, clustering and optionally defined channels.
        Saves plot in :attr:`ModelComparator.save_dir`.

        Parameters
        ----------
        exp_names
            Experiments for which to plot a UMAP.
        channels
            Channels that should be visualised.
        save_prefix
            Plot is saved under `{save_prefix}umap_{exp_name}.png`.

        Returns
        -------
        Nothing, plots and saves UMAP
        """
        from campa.tl._cluster import add_clustering_to_adata

        if channels is None:
            channels = []
        if exp_names is None:
            exp_names = self.exps.keys()
        for exp_name in exp_names:
            # prepare adata, add clustering with correct colors
            adata = self.mpps[exp_name].get_adata(obsm={"X_umap": "umap"})
            assert self.mpps[exp_name].data_dir is not None, "need to have data_dir defined to add clustering"
            cluster_name = self.exps[exp_name].config["cluster"]["cluster_name"]
            cluster_annotation = self.exps[exp_name].get_split_cluster_annotation(cluster_name)
            add_clustering_to_adata(
                self.mpps[exp_name].data_dir, cluster_name, adata, cluster_annotation  # type: ignore[arg-type]
            )

            # plot umap
            conditions = [process_condition_desc(c)[0] for c in self.exps[exp_name].data_params["condition"]]
            sc.pl.umap(
                adata,
                color=conditions + [cluster_name] + channels,
                vmax="p99",
                show=False,
            )
            plt.suptitle(exp_name)
            if self.save_dir is not None:
                plt.savefig(
                    os.path.join(self.save_dir, f"{save_prefix}umap_{exp_name}.png"),
                    dpi=100,
                )
