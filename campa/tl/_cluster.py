from __future__ import annotations

from typing import (
    Any,
    Mapping,
    Iterable,
    TYPE_CHECKING,
    MutableMapping,
)

if TYPE_CHECKING:
    from campa.tl import Experiment
    import anndata as ad

from copy import deepcopy
import os
import json
import pickle
import urllib
import logging

from pynndescent import NNDescent
from matplotlib.colors import rgb2hex
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from campa.data import MPPData
from campa.utils import merged_config
from campa.constants import campa_config
from campa.tl._evaluate import Predictor


# annotation fns
def annotate_clustering(
    clustering: Iterable[str | int],
    annotation: pd.DataFrame,
    cluster_name: str,
    annotation_col: str | None = None,
) -> Iterable[str | int]:
    """
    Annotate clustering according to annotation.

    Parameters
    ----------
    clustering
        Clustering to annotate.
    annotation
        Annotation table containing mapping from clustering values to annotated values.
    cluster_name
        Column in ``annotation`` containing current clustering values.
    annotation_col
        Columns in ``annotation`` containing desired annotation.

    Returns
    -------
    Iterable[str, int]
        Annotated clustering.
    """
    if annotation_col is None:
        return clustering
    if cluster_name == annotation_col:
        return clustering
    return np.array(annotation.set_index(cluster_name)[annotation_col].loc[clustering])


def add_clustering_to_adata(
    data_dir: str,
    cluster_name: str,
    adata: ad.AnnData,
    annotation: pd.DataFrame,
    added_name: str = None,
    annotation_col: str = None,
) -> None:
    """
    Add clustering to adata.

    Adds `cluster_name` to `adata.obs[added_name]` and colour values for each cluster stored in the annotation dict.
    Assumes that adata has the same dimensionality as the clustering.

    Parameters
    ----------
    data_dir
        Full path to directory containing ``{cluster_name}.npy``.
    cluster_name
        Column in ``annotation`` containing current clustering values.
    adata
        Adata to which to add the clustering to.
    annotation
        Annotation table containing mapping from clustering values to annotated values.
    added_name
        Name to use for clustering in adata. Defaults to `annotation_col`
    annotation_col
        Columns in ``annotation`` containing desired annotation. Defaults to `cluster_name`.

    Returns
    -------
    Nothing, adds clustering to `adata.obs[added_name]`.
    """
    if added_name is None:
        added_name = annotation_col
        if added_name is None:
            added_name = cluster_name
    if annotation_col is None:
        annotation_col = cluster_name
    # load clustering
    clustering = np.load(os.path.join(data_dir, f"{cluster_name}.npy"), allow_pickle=True)
    # map clustering to annotation
    clustering = annotate_clustering(clustering, annotation, cluster_name, annotation_col)
    # add to adata
    adata.obs[added_name] = clustering
    adata.obs[added_name] = adata.obs[added_name].astype("category")
    # add cmap
    cmap = annotation.drop_duplicates(subset=annotation_col).set_index(annotation_col)[annotation_col + "_colors"]
    adata.uns[added_name + "_colors"] = list(cmap.loc[adata.obs[added_name].cat.categories])


class Cluster:
    """
    Cluster data.

    Contains functions to create a (subsampled) :class:`MPPData` for clustering, cluster it,
    and to project the clustering to other MPPDatas.

    Cluster is initialised from a cluster config dictionary with the following keys:

    - ``data_config``: name of the data config to use, should be registered in ``campa.ini``
    - ``data_dirs``: where to read data from (relative to ``DATA_DIR`` defined in data config)
    - ``process_like_dataset``: name of dataset that gives parameters for processing (except subsampling/subsetting)
    - ``subsample``: (bool) subsampling of pixels
    - ``subsample_kwargs``: kwargs for :meth:`MPPData.subsample` defining the fraction of pixels to be sampled
    - ``subset``: (bool) subset to objects with certain metadata.
    - ``subset_kwargs``: kwargs to :meth:`MPPData.subset` defining which object to subset to
    - ``seed``: random seed to make subsampling reproducible
    - ``cluster_data_dir``: name of the dir containing the mpp_data that is clustered. Relative to EXPERIMENT_DIR
    - ``cluster_name``: name of the cluster assignment file
    - ``cluster_rep``: representation that should be clustered
      (name of existing file, should be predicted with :meth:`Predictor.get_representation`).
    - ``cluster_method``: `leiden` or `kmeans` (`kmeans` not tested).
    - ``leiden_resolution``: resolution parameter for leiden clustering.
    - ``kmeans_n``: number of clusters for `kmeans`.
    - ``umap``: (bool) predict UMAP of ``cluster_rep``.

    Parameters
    ----------
    config
        Cluster config.
    cluster_mpp
        Data to cluster.
    save_config
        Save cluster config in ``{config['cluster_data_dir']}/cluster_params.json``.
    """

    config: MutableMapping[str, Any] = {
        # --- cluster data creation (mpp_data is saved to cluster_data_dir) ---
        "data_config": "NascentRNA",
        "data_dirs": [],
        # name of dataset that gives params for processing (except subsampling/subsetting)
        "process_like_dataset": None,
        "subsample": False,
        "subsample_kwargs": {},
        "subset": False,
        "subset_kwargs": {},
        "seed": 42,
        # name of the dir containing the mpp_data that is clustered. Relative to campa_config.EXPERIMENT_DIR
        "cluster_data_dir": None,
        # --- cluster params ---
        # name of the cluster assignment file
        "cluster_name": "clustering",
        # representation that should be clustered (name of existing file)
        "cluster_rep": "latent",
        "cluster_method": "leiden",  # leiden or kmeans
        "leiden_resolution": 0.8,
        "kmeans_n": 20,
        "umap": True,  # calculate umap of cluster data
    }

    def __init__(self, config: Mapping[str, Any], cluster_mpp: MPPData = None, save_config: bool = False):
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = merged_config(self.config, config)
        """Cluster config."""

        self.data_config_name = self.config["data_config"]
        self.data_config = campa_config.get_data_config(self.data_config_name)
        # load dataset_params
        self.dataset_params = None
        if self.config["process_like_dataset"] is not None:
            params_fname = os.path.join(
                self.data_config.DATASET_DIR,
                self.config["process_like_dataset"],
                "params.json",
            )
            self.dataset_params = json.load(open(params_fname))
        # initialise cluster_mpp
        self._cluster_mpp = cluster_mpp
        # try to load it from disk if not already initialised
        self.cluster_mpp

        # initialise annotation
        self._cluster_annotation: pd.DataFrame | None = None
        self.cluster_annotation

        # save config
        if save_config:
            config_fname = os.path.join(
                campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"], "cluster_params.json"
            )
            os.makedirs(os.path.dirname(config_fname), exist_ok=True)
            json.dump(self.config, open(config_fname, "w"), indent=4)

    @classmethod
    def from_cluster_data_dir(cls, data_dir: str) -> Cluster:
        """
        Initialise from existing ``cluster_data_dir``.

        Parameters
        ----------
        data_dir
            Dir containing complete :class:`MPPData` with ``cluster_rep`` and ``cluster_name`` files.
            Relative to ``campa_config.EXPERIMENT_DIR``.
        """
        # load mpp_data and cluster_params (for reference) from data_dir
        # TODO
        config_fname = os.path.join(campa_config.EXPERIMENT_DIR, data_dir, "cluster_params.json")
        config = json.load(open(config_fname))
        return cls(config, save_config=False)

    @classmethod
    def from_exp(
        cls, exp: Experiment, cluster_config: Mapping[str, Any] | None = None, data_dir: str = None
    ) -> Cluster:
        """
        Initialise from experiment for clustering of entire data that went into creating training data.

        Cluster parameters are read from ``Experiment.config['cluster']``.

        Parameters
        ----------
        exp
            Trained Experiment.
        cluster_config
            Additional cluster parameters (like subsampling etc). Overwrites default cluster parameters.
        data_dir
            Directory containing ``cluster_mpp`` (relative to ``{exp.dir}/{exp.name}``).
        """
        add_cluster_config = cluster_config if cluster_config is not None else {}
        cur_cluster_config = deepcopy(exp.config["cluster"])
        # add information on data to cluster
        cur_cluster_config["data_config"] = exp.config["data"]["data_config"]
        cur_cluster_config["data_dirs"] = exp.data_params["data_dirs"]
        # only process data for model if experiment has model to evaluate
        cur_cluster_config["process_like_dataset"] = exp.config["data"]["dataset_name"]
        cur_cluster_config["seed"] = exp.data_params["seed"]
        if data_dir is None:
            data_dir = os.path.join("aggregated", "sub-" + cur_cluster_config["subsample_kwargs"]["frac"])
        cur_cluster_config["cluster_data_dir"] = os.path.join(exp.dir, exp.name, data_dir)
        # add passed cluster_config
        cur_cluster_config = merged_config(cur_cluster_config, add_cluster_config)
        return cls(cur_cluster_config, save_config=True)

    @classmethod
    def from_exp_split(cls, exp: Experiment) -> Cluster:
        """
        Initialise from experiment for clustering of val/test split.

        Parameters
        ----------
        exp
            Trained Experiment.
        """
        # TODO load exp
        # est = Estimator(exp.set_to_evaluate())
        # mpp_data = est.ds.data[exp.evaluate_config['split']]

        cluster_config = deepcopy(exp.config["cluster"])
        # add data_config
        cluster_config["data_config"] = exp.config["data"]["data_config"]
        cluster_config["subsample"] = False
        cluster_config["cluster_data_dir"] = os.path.join(
            exp.dir,
            exp.name,
            f"results_epoch{exp.epoch:03d}",
            exp.evaluate_config["split"],
        )

        return cls(cluster_config, save_config=True)

    def set_cluster_name(self, cluster_name):
        """
        Change the cluster name and reloads ``cluster_mpp``, and ``cluster_annotation``.
        """
        if self.config["cluster_name"] != cluster_name:
            self.config["cluster_name"] = cluster_name
            self._cluster_mpp = self._load_cluster_mpp()
            self._cluster_annotation = self._load_cluster_annotation()

    # --- Properties and loading fns ---
    @property
    def cluster_mpp(self) -> MPPData | None:
        """
        :class:`MPPData` that is used for clustering.

        None if data could not be loaded.
        """
        if self._cluster_mpp is None:
            self._cluster_mpp = self._load_cluster_mpp()
        return self._cluster_mpp

    def _load_cluster_mpp(self) -> MPPData | None:
        """
        Load MPPData that is used for clustering.

        Tries to read MPPData with cluster_rep and cluster_name from cluster_data_dir.

        Returns
        -------
        :class:`MPPData` or None if data could not be loaded.
        """
        data_dir = self.config["cluster_data_dir"]
        rep = self.config["cluster_rep"]
        name = self.config["cluster_name"]
        # check that dir is defined
        if data_dir is None:
            self.log.warning("Cluster_data_dir is None, cannot load cluster_mpp")
            return None
        # load data
        try:
            mpp_data = MPPData.from_data_dir(
                data_dir,
                base_dir=campa_config.EXPERIMENT_DIR,
                optional_keys=["mpp", rep, name, "umap"],
                data_config=self.data_config_name,
            )
            self.log.info(f"Loaded cluster_mpp {mpp_data}")
            return mpp_data
        except FileNotFoundError:
            self.log.warning(f"Could not load MPPData from {data_dir}")
            return None

    @property
    def cluster_annotation(self) -> pd.DataFrame:
        """
        Cluster annotation `pd.DataFrame`, read from ``{cluster_name}_annotation.csv``.
        """
        if self._cluster_annotation is None:
            self._cluster_annotation = self._load_cluster_annotation()
        return self._cluster_annotation

    def _load_cluster_annotation(self, recreate: bool = False) -> pd.DataFrame:
        """
        Read cluster annotation file / create it.
        """
        fname = os.path.join(
            campa_config.EXPERIMENT_DIR,
            self.config["cluster_data_dir"],
            f"{self.config['cluster_name']}_annotation.csv",
        )
        # try to read file
        if os.path.exists(fname) and not recreate:
            annotation = pd.read_csv(fname, index_col=0, dtype=str, keep_default_na=False)
            return annotation
        else:
            # can create?
            if (self.cluster_mpp is None) or (self.cluster_mpp.data(self.config["cluster_name"]) is None):
                self.log.info("cannot create annotation without clustering in cluster_mpp")
                return None
            # create empty annnotation from unique clusters and empty cluster for background in images
            annotation = pd.DataFrame(
                {
                    self.config["cluster_name"]: sorted(
                        np.unique(self.cluster_mpp.data(self.config["cluster_name"])),
                        key=int,
                    )
                    + [""]
                }
            )
            annotation.index.name = "index"
            # save annotation
            annotation.to_csv(fname)
            self._cluster_annotation = annotation
            # add colors
            self.add_cluster_colors(colors=None)
            return self._cluster_annotation

    # --- fns modifying annotation ---
    def add_cluster_annotation(
        self,
        annotation: Mapping[str, str],
        to_col: str,
        from_col: str | None = None,
        colors: Mapping[str, str] | None = None,
    ) -> None:
        """
        Add annotation and colormap to clustering.

        Is saved in ``{cluster_name}_annotation.csv``

        Parameters
        ----------
        annotation:
            Dict mapping from values of ``from_col`` to the annotation.
        to_col
            Name under which the annotation should be saved.
        from_col
            Optionally set the annotation name from which to annotate. Default is ``cluster_name``.
        colors
            Colour dict, mapping from annotations to hex colours. Default is using tab20 colormap.
        """
        if from_col is None:
            from_col = self.config["cluster_name"]
        df = pd.DataFrame.from_dict(annotation, orient="index", columns=[to_col])
        # remove to_col and to_col_colors if present
        annotation = self.cluster_annotation
        annotation.drop(columns=[to_col], errors="ignore", inplace=True)
        # add annotation col
        annotation = pd.merge(annotation, df, how="left", left_on=from_col, right_index=True)
        self._cluster_annotation = annotation
        # save to disk
        fname = os.path.join(
            campa_config.EXPERIMENT_DIR,
            self.config["cluster_data_dir"],
            f"{self.config['cluster_name']}_annotation.csv",
        )
        self._cluster_annotation.to_csv(fname)
        # add colors
        self.add_cluster_colors(colors, to_col)

    def add_cluster_colors(self, colors: Mapping[str, str] | None, from_col: str | None = None) -> None:
        """
        Add colours to clustering or to annotation.

        Adds column ``{from_col}_colors`` to :attr:`Cluster.cluster_annotation`
        and saves it to ``{cluster_name}_annotation.csv``.

        Parameters
        ----------
        colors
            Colour dict, mapping from unique clustering values from ``from_col`` to hex colours.
            Default is using tab20 colormap.
        from_col
            Optionally set clustering name for which to add colours. Default is ``cluster_name``.
        """
        if from_col is None:
            from_col = self.config["cluster_name"]
        to_col = from_col + "_colors"
        # remove previous colors
        annotation = self.cluster_annotation
        annotation.drop(columns=[to_col], errors="ignore", inplace=True)
        # add colors
        if colors is None:
            # get unique values, removing nan and empty string
            values = list(np.unique(annotation[from_col].dropna()))
            if "" in values:
                values.remove("")
            N = len(values)
            cmap = plt.get_cmap("tab20", N)
            colors = {k: rgb2hex(cmap(i)) for i, k in enumerate(values)}
        df = pd.DataFrame.from_dict(colors, orient="index", columns=[to_col])
        annotation = pd.merge(annotation, df, how="left", left_on=from_col, right_index=True)
        # fill nan values with white background color
        annotation[to_col] = annotation[to_col].fillna("#ffffff")
        self._cluster_annotation = annotation
        # save to disk
        fname = os.path.join(
            campa_config.EXPERIMENT_DIR,
            self.config["cluster_data_dir"],
            f"{self.config['cluster_name']}_annotation.csv",
        )
        self._cluster_annotation.to_csv(fname)

    # --- getters ---
    def get_nndescent_index(self, recreate=False):
        """
        Calculate and return pynndescent index of existing clustering for fast prediction of new data.
        """
        index_fname = os.path.join(
            campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"], "pynndescent_index.pickle"
        )
        if os.path.isfile(index_fname) and not recreate:
            try:
                # load and return index
                return pickle.load(open(index_fname, "rb"))
            except AttributeError as e:
                if "'NNDescent' object has no attribute 'parallel_batch_queries'" in str(e):
                    # need to recreate index
                    self.log.info(
                        "Version of pynndescent does not match version that index was created with. Recreating index."
                    )
                else:
                    raise (e)
        # need to create index
        # check that cluster_rep has been computed already for cluster_mpp
        assert self.cluster_mpp is not None
        assert self.cluster_mpp.data(self.config["cluster_rep"]) is not None
        self.log.info(f"Creating pynndescent index for {self.config['cluster_rep']}")

        data = self.cluster_mpp._data[self.config["cluster_rep"]]
        if self.config["cluster_rep"] == "mpp":
            data = self.cluster_mpp.center_mpp
        index = NNDescent(data.astype(np.float32))
        pickle.dump(index, open(index_fname, "wb"))
        return index

    # --- functions creating and adding data to cluster_mpp ---
    def create_cluster_mpp(self):
        """
        Use cluster parameters to create and save :attr:`Cluster.cluster_mpp` to use for clustering.

        Raises
        ------
        ValueError if config does not contain keys ``data_dirs`` and ``process_like_dataset``.
        """
        # TODO: add option how to process data (e.g. for MPPcluster, do not need to add neighborhood)
        self.log.info("creating cluster mpp from config")
        # check that have required information
        if len(self.config["data_dirs"]) == 0:
            raise ValueError("Cannot create cluster data without data_dirs")
        self.log.info(f"processing cluster_mpp like dataset {self.config['process_like_dataset']}")
        # load params to use when processing
        data_config = campa_config.get_data_config(self.config["data_config"])
        data_params = json.load(
            open(
                os.path.join(
                    data_config.DATASET_DIR,
                    self.config["process_like_dataset"],
                    "params.json",
                ),
            )
        )
        # load and process data
        mpp_data = []
        for data_dir in self.config["data_dirs"]:
            mpp_data.append(
                MPPData.from_data_dir(
                    data_dir,
                    seed=self.config["seed"],
                    data_config=self.config["data_config"],
                )
            )
            # subset if necessary (do before subsampling, to get expected # of samples)
            if data_params["subset"]:
                mpp_data[-1].subset(**data_params["subset_kwargs"])
        mpp_data = MPPData.concat(mpp_data)
        # TODO after have reproduced data, could do subsampling inside for loop (after subsetting)
        if self.config["subsample"]:
            mpp_data = mpp_data.subsample(
                add_neighborhood=data_params["neighborhood"],
                neighborhood_size=data_params["neighborhood_size"],
                **self.config["subsample_kwargs"],
            )
        mpp_data.prepare(data_params)

        self._cluster_mpp = mpp_data
        if self.config["cluster_data_dir"] is not None:
            self._cluster_mpp.write(os.path.join(campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"]))

    def predict_cluster_rep(self, exp: Experiment) -> None:
        """
        Use experiment to predict the necessary cluster representation.

        Saves predicted representations to ``cluster_data_dir``.

        Parameters
        ----------
        exp
            Experiment to use for predicting cluster_rep.
        """
        assert self.cluster_mpp is not None
        if self.cluster_mpp.data(self.config["cluster_rep"]) is not None:
            self.log.info(f"cluster_mpp already contains key {self.config['cluster_rep']}. Not recalculating.")
            return
        pred = Predictor(exp)
        cluster_rep = pred.get_representation(self.cluster_mpp, rep=self.config["cluster_rep"])
        self.cluster_mpp._data[self.config["cluster_rep"]] = cluster_rep
        # save cluster_rep
        if self.config["cluster_data_dir"] is not None:
            self.cluster_mpp.write(
                os.path.join(campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"]),
                save_keys=[self.config["cluster_rep"]],
            )

    def create_clustering(self) -> None:
        """
        Cluster :attr:`Cluster.cluster_mpp` using ``cluster_method`` defined in :attr:`Cluster.config`.

        If ``cluster_data_dir`` is defined, saves clustering there.

        Raises
        ------
        ValueError if cluster_rep is not available
        """
        # check that have cluster_rep
        assert self.cluster_mpp is not None
        if self.cluster_mpp.data(self.config["cluster_rep"]) is None:
            raise ValueError(f"Key {self.config['cluster_rep']} is not available for clustering.")
        save_keys = [self.config["cluster_name"]]
        # cluster
        if self.config["cluster_method"] == "leiden":
            self.log.info("Creating leiden clustering")
            # leiden clustering
            adata = self.cluster_mpp.get_adata(X=self.config["cluster_rep"])
            sc.pp.neighbors(adata)
            sc.tl.leiden(
                adata,
                resolution=self.config["leiden_resolution"],
                key_added="clustering",
            )
            self.cluster_mpp._data[self.config["cluster_name"]] = np.array(adata.obs["clustering"])
            if self.config["umap"]:
                self.log.info("Calculating umap")
                sc.tl.umap(adata)
                self.cluster_mpp._data["umap"] = adata.obsm["X_umap"]
                save_keys.append("umap")
        elif self.config["cluster_method"] == "kmeans":
            self.log.info("Creating kmeans clustering")
            from sklearn.cluster import KMeans

            est = KMeans(n_clusters=self.config["kmeans_n"], random_state=0)
            kmeans = est.fit(self.cluster_mpp.data(self.config["cluster_rep"])).labels_
            # TODO: cast kmeans to str?
            self.cluster_mpp._data[self.config["cluster_name"]] = kmeans
        else:
            raise NotImplementedError(self.config["cluster_method"])

        # create and save pynndescent index
        _ = self.get_nndescent_index(recreate=True)

        # save to cluster_data_dir
        if self.config["cluster_data_dir"] is not None:
            self.cluster_mpp.write(
                os.path.join(campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"]),
                save_keys=save_keys,
            )

        # add umap if not exists already
        if self.config["umap"] and not self.config["cluster_method"] == "leiden":
            self.add_umap()

        # recreate annotation file
        self._cluster_annotation = self._load_cluster_annotation(recreate=True)

    def add_umap(self) -> None:
        """
        If umap does not yet exist, but should be calculated, calculates umap.
        """
        assert self.cluster_mpp is not None
        if self.config["umap"]:
            if self.cluster_mpp.data("umap") is not None:
                # already have umap, no need to calculate
                self.log.info("found existing umap, not recalculating.")
                return
            self.log.info("Calculating umap")
            adata = self.cluster_mpp.get_adata(X=self.config["cluster_rep"])
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
            self.cluster_mpp._data["umap"] = adata.obsm["X_umap"]
            # save to cluster_data_dir
            if self.config["cluster_data_dir"] is not None:
                self.cluster_mpp.write(
                    os.path.join(campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"]),
                    save_keys=["umap"],
                )

    # --- using existing cluster_mpp, project clustering ---
    def project_clustering(self, mpp_data: MPPData, save_dir: str | None = None, batch_size: int = 200000) -> MPPData:
        """
        Project already computed clustering from :attr:`Cluster.cluster_mpp` to ``mpp_data``.

        Parameters
        ----------
        mpp_data
            Data to project the clustering to. Should contain ``cluster_rep``.
        save_dir
            Full path to dir where the clustering should be saved.
        batch_size
            Iterate over data in batches of size ``batch_size``.

        Returns
        -------
        :class:`MPPData`
            Data with clustering.
        """
        # check that clustering has been computed already for cluster_mpp
        assert self.cluster_mpp is not None
        assert self.cluster_mpp.data(self.config["cluster_name"]) is not None
        assert mpp_data.data(self.config["cluster_rep"]) is not None

        # get NNDescent index for fast projection
        index = self.get_nndescent_index()
        self.log.info(f"Projecting clustering to {len(mpp_data.obj_ids)} samples")

        # func for getting max count cluster in each row
        def most_frequent(arr):
            els, counts = np.unique(arr, return_counts=True)
            return els[np.argmax(counts)]

        # project clusters
        clustering = []
        samples = mpp_data._data[str(self.config["cluster_rep"])]
        if self.config["cluster_rep"] == "mpp":  # use center mpp in this special case
            samples = mpp_data.center_mpp
        for i in np.arange(0, samples.shape[0], batch_size):
            self.log.info(f"processing chunk {i}")
            cur_samples = samples[i : i + batch_size]
            neighs = index.query(cur_samples.astype(np.float32), k=15)[0]
            # NOTE: do not use apply_along_axis, because dtype is inferred incorrectly!
            clustering.append(
                np.array(
                    [most_frequent(row) for row in self.cluster_mpp._data[self.config["cluster_name"]][neighs]],
                    dtype=self.cluster_mpp._data[self.config["cluster_name"]].dtype,
                )
            )
        # convert from str to int to save space when saving full data NOTE new
        mpp_data._data[self.config["cluster_name"]] = np.concatenate(clustering)

        # save
        if save_dir is not None:
            mpp_data.write(save_dir, save_keys=[self.config["cluster_name"]])
        return mpp_data

    def predict_cluster_imgs(self, exp: Experiment) -> MPPData | None:
        """
        Predict cluster images from experiment.

        Parameters
        ----------
        exp
            Experiment.
        """
        if exp.is_trainable:
            # create Predictor
            pred = Predictor(exp)
            # predict clustering on imgs
            img_save_dir = os.path.join(
                exp.full_path,
                f"results_epoch{pred.est.epoch:03d}",
                exp.config["evaluation"]["split"] + "_imgs",
            )
            mpp_imgs = pred.est.ds.imgs[exp.config["evaluation"]["split"]]
            # add latent space to mpp_imgs + subset
            try:
                mpp_imgs.add_data_from_dir(
                    img_save_dir,
                    keys=[self.config["cluster_rep"]],
                    subset=True,
                    base_dir="",
                )
            except FileNotFoundError:
                self.log.warning(
                    f"Did not find {self.config['cluster_rep']} in {img_save_dir}. Run create_clustering first."
                )
                return None
        else:
            img_save_dir = os.path.join(
                exp.full_path,
                "results_epoch000",
                exp.config["evaluation"]["split"] + "_imgs",
            )
            mpp_imgs = MPPData.from_data_dir(img_save_dir, base_dir="", data_config=self.data_config_name)
        self.log.info(f"Projecting cluster_imgs for {exp.dir}/{exp.name}")
        return self.project_clustering(mpp_imgs, save_dir=img_save_dir)

    # --- use cluster_mpp to query HPA ---
    def get_hpa_localisation(
        self,
        cluster_name: str = "clustering_res0.5",
        thresh: float = 1,
        limit_to_groups: Mapping[str, str | list[str]] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Mapping[str, Any]]:
        """
        Query subcellular localisation for each cluster from Human Protein Atlas (https://www.proteinatlas.org).

        Calculates cluster loadings and returns the subcellular localisations of the channels that are enriched for each cluster.
        Requires "hpa_gene_name" column in channel_metadata.csv file in DATA_DIR to map channel names to genes available in HPA.

        Parameters
        ----------
        cluster_name
            Clustering to calculate localisations for. Must exist already.
        thresh
            Minimum z-scored intensity value of channel in cluster to be considered for HPA query.
            thresh=0 considers all enriched channel of this cluster
        limit_to_groups
            Dict with obs as keys and groups from obs as values, to subset data before calculating loadings.
        kwargs
            Keyword arguments for :func:`campa.tl.query_hpa_subcellular_location`.

        Returns
        -------
            Mapping[str, Mapping[str, Any]]:
                Results dictionary with clusters as keys, and return value from :func:`campa.tl.query_hpa_subcellular_location`
        """
        self.set_cluster_name(cluster_name)
        adata = self.cluster_mpp.get_adata(X="mpp", obsm={"X_latent": "latent", "X_umap": "umap"})
        # ensure that clustering is available in adata
        add_clustering_to_adata(
            os.path.join(campa_config.EXPERIMENT_DIR, self.config["cluster_data_dir"]),
            cluster_name,
            adata,
            self.cluster_annotation,
        )

        # subset data
        if limit_to_groups is None:
            limit_to_groups = {}
        for key, groups in limit_to_groups.items():
            if not isinstance(groups, list):
                groups = [groups]
            adata = adata[adata.obs[key].isin(groups)]

        # calculate mean z-scored intensity values
        means = (
            pd.concat(
                [
                    pd.DataFrame(adata.X, columns=adata.var_names).reset_index(drop=True),
                    adata.obs[[cluster_name]].reset_index(drop=True),
                ],
                axis=1,
            )
            .groupby(cluster_name)
            .aggregate("mean")
        )
        means = (means - means.mean()) / means.std()
        # means = means.apply(zscore, axis=1)

        # for each cluster, determine channels that localise to this cluster (mean z-scored intensity > thresh)
        # and map channel names to gene names used in HPA
        channels_metadata = pd.read_csv(os.path.join(self.data_config.DATA_DIR, "channels_metadata.csv"), index_col=0)
        cluster_localisation = {}
        for idx, row in means.iterrows():
            channels = list(row[row > thresh].index)
            weights = list(row[row > thresh])
            cluster_localisation[idx] = (list(channels_metadata.set_index("name").loc[channels].gene_name_hpa), weights)

        # query hpa for each cluster
        results = {}
        for idx, (genes, weights) in cluster_localisation.items():
            results[idx] = query_hpa_subcellular_location(genes, gene_weights=weights, **kwargs)
        return results


def query_hpa_subcellular_location(
    genes: list[str], gene_weights: Iterable[float] | None = None, filter_reliability: Iterable[str] = ["Uncertain"]
) -> Mapping[str, Any]:
    """
    Query the Human Protein Atlas for a consensus subcellular locations from a list of genes.

    HPA is availablen at https://www.proteinatlas.org

    Parameters
    ----------
    genes
        List of genes to query in the HPA (using field "gene_name").
    gene_weights
        List of weights for each gene, used to compute main subcellular locations.
    filter_reliability
        Do not return genes with this subcellular location reliability ("Reliability IF").
        Available reliabilities (in order from most reliable to least reliable) are: Enhanced, Supported, Approved, Uncertain.
        See also https://www.proteinatlas.org/about/assays+annotation#if_reliability_score

    Returns
    -------
        Mapping[str, Any]:
            Results dictionary with keys:
                - hpa_data: data frame of available genes and their subcellular locations according to HPA data.
                - subcellular_locations: pd.Series of all subcellular locations ocurring for this list of genes, sorted by most frequent
    """
    if gene_weights is None:
        gene_weights = [1] * len(genes)
    url_str = "http://www.proteinatlas.org/api/search_download.php?search=gene_name:{gene}&format=json&columns=g,gs,scml,scal,relce&compress=no"
    data = []
    index = []
    for gene in genes:
        if gene is None or gene == np.nan:
            continue
        cur_url_str = url_str.format(gene=gene)
        with urllib.request.urlopen(cur_url_str) as url:
            res = json.load(url)
            if len(res) > 0:
                index.append(gene)
                data.append(res[0])
            else:
                print(f"No result for {gene}")
    data = pd.DataFrame(data, index=index)
    # filter out any columns with filter_reliability or None reliability score
    data = data[~data["Reliability (IF)"].isin(filter_reliability + [None])]
    data = pd.merge(
        data, pd.DataFrame({"gene_weights": gene_weights}, index=genes), how="left", right_index=True, left_index=True
    )
    # summarise locations & their occurrence counts
    summary = {}
    for locations, weight in zip(data["Subcellular main location"], data["gene_weights"]):
        for loc in locations:
            if loc not in summary.keys():
                summary[loc] = weight
            else:
                summary[loc] += weight
    summary = pd.Series(summary).sort_values(ascending=False)
    return {"hpa_data": data, "subcellular_locations": summary}


def prepare_full_dataset(
    experiment_dir: str, save_dir: str = "aggregated/full_data", data_dirs: list[str] | None = None
) -> None:
    """
    Prepare all data for clustering by predicting cluster-rep.

    Parameters
    ----------
    experiment_dir
        Experiment directory relative to ``EXPERIMENT_PATH``.
    save_dir
        Directory to save prepared full data to, relative to ``experiment_dir``.
    data_dirs
        Data to prepare. Defaults for ``exp.data_params['data_dirs']``.
    """
    from campa.tl import Experiment

    log = logging.getLogger("prepare full dataset")
    exp = Experiment.from_dir(experiment_dir)
    # iterate over all data dirs
    if data_dirs is None:
        data_dirs = exp.data_params["data_dirs"]
    print("iterating over data dirs", data_dirs)
    for data_dir in data_dirs:
        log.info(f"Processing data_dir {data_dir}")
        mpp_data = MPPData.from_data_dir(
            data_dir,
            data_config=exp.config["data"]["data_config"],
        )
        # params for partial saving of mpp_data
        mpp_params = {"base_data_dir": data_dir, "subset": True}
        # prepare mpp_data
        log.info("Preparing data")
        mpp_data.prepare(exp.data_params)
        if exp.config["cluster"]["cluster_rep"] == "mpp":
            # just save mpp
            mpp_data.write(
                os.path.join(exp.full_path, save_dir, data_dir),
                mpp_params=mpp_params,
                save_keys=["mpp"],
            )
        else:
            # need to predict rep - prepare neighborhood
            if exp.data_params["neighborhood"]:
                mpp_data.add_neighborhood(exp.data_params["neighborhood_size"])
            # predict rep
            log.info("Predicting latent")
            pred = Predictor(exp)
            pred.predict(
                mpp_data,
                reps=[exp.config["cluster"]["cluster_rep"]],
                save_dir=os.path.join(exp.full_path, save_dir, data_dir),
                mpp_params=mpp_params,
            )


def create_cluster_data(
    experiment_dir: str,
    subsample: bool = False,
    frac: float = 0.005,
    save_dir: str | None = None,
    cluster: bool = False,
) -> None:
    """
    Create (subsampled) data for clustering.

    Uses dataset used to train experiment.

    Parameters
    ----------
    experiment_dir
        Experiment directory relative to ``EXPERIMENT_PATH``.
    subsample
        Subsample the data.
    frac
        Fraction of pixels to use for clustering if ``subsample`` is True.
    save_dir
        Directory to save subsampled cluster data, relative to ``experiment_dir``.
        Default is ``aggregated/sub-FRAC``.
    cluster
        Use cluster parameters in Experiment config to cluster the subsetted data.
    """
    from campa.tl import Experiment

    exp = Experiment.from_dir(experiment_dir)
    cluster_config = {
        "subsample": subsample,
        "subsample_kwargs": {"frac": frac},
    }
    save_dir = save_dir if save_dir is not None else f"aggregated/sub-{frac}"
    cl = Cluster.from_exp(exp, cluster_config=cluster_config, data_dir=save_dir)
    # create cluster_mpp
    cl.create_cluster_mpp()
    # predict rep
    cl.predict_cluster_rep(exp)
    if cluster:
        # cluster (also gets umap)
        cl.create_clustering()
    else:
        # get umap
        cl.add_umap()


def project_cluster_data(
    experiment_dir: str,
    cluster_data_dir: str,
    cluster_name: str = "clustering",
    save_dir: str = "aggregated/full_data",
    data_dir: str | None = None,
) -> None:
    """
    Project existing clustering to new data.

    Parameters
    ----------
    experiment_dir
        Experiment directory relative to ``EXPERIMENT_PATH``.
    cluster_data_dir
        Directory in which clustering is stored relative to experiment dir. Usually in ``aggregated/sub-FRAC``.
    cluster_name
        Name of clustering to project.
    save_dir
        Directory in which the data to be projected is stored, relative to ``experiment_dir``.
    data_dir
        Data directory to project. If not specified, project all ``data_dir``s in ``save_dir``.
        Relative to ``save_dir``.
    """
    from campa.tl import Experiment

    exp = Experiment.from_dir(experiment_dir)
    # set up cluster data
    cl = Cluster.from_cluster_data_dir(os.path.join(exp.dir, exp.name, cluster_data_dir))
    cl.set_cluster_name(cluster_name)
    assert cl.cluster_mpp is not None
    assert cl.cluster_mpp.data(cluster_name) is not None, f"cluster data needs to contain clustering {cluster_name}"
    # iterate over all data dirs
    data_dirs: Iterable[str] = exp.data_params["data_dirs"] if data_dir is None else [data_dir]
    for cur_data_dir in data_dirs:
        # load mpp_data with cluster_rep
        mpp_data = load_full_data_dict(
            exp, keys=["x", "y", "obj_ids", cl.config["cluster_rep"]], data_dirs=[cur_data_dir], save_dir=save_dir
        )[cur_data_dir]
        cl.project_clustering(mpp_data, save_dir=os.path.join(exp.full_path, save_dir, cur_data_dir))


def load_full_data_dict(
    exp: Experiment,
    keys: Iterable[str] = ("x", "y", "obj_ids", "latent"),
    data_dirs: Iterable[str] | None = None,
    save_dir: str = "aggregated/full_data",
) -> Mapping[str, MPPData]:
    """
    Load mpp_datas used in experiment in a dict.

    NOTE: this might take a few minutes, as all data needs to be loaded.

    Parameters
    ----------
    exp
        Experiment from which to load the mpp_datas.
    keys
        Controls which numpy data matrices are being loaded. Passed to :meth:`MPPData.from_data_dir`,
        with `optional_keys` set to empty list. Excluding mpp here speeds up loading.
    data_dirs
        Directories that should be loaded, if None, all ``data_dirs`` are loaded.
    save_dir
        Directory in which the data to be loaded is stored, relative to ``{exp.dir}/{exp.name}``.


    Returns
    -------
    Dictionary with `data_dirs` as keys and :class:`MPPData` as values.
    """
    if data_dirs is None:
        data_dirs = exp.data_params["data_dirs"]
    mpp_datas = {}
    for data_dir in data_dirs:
        mpp_datas[data_dir] = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(exp.full_path, save_dir),
            keys=keys,
            optional_keys=[],
            data_config=exp.config["data"]["data_config"],
        )
    return mpp_datas


def get_clustered_cells(
    mpp_datas: Mapping[str, MPPData], cl: Cluster, cluster_name: str, num_objs: int = 5
) -> Mapping[str, Mapping[str, MPPData]]:
    """
    Get `num_objs` example cells for each `mpp_data`.

    Parameters
    ----------
    mpp_datas
        Data from which to sample and cluster cells.
    cl
        Clustering object to use.
    cluster_name
        Name of the clustering.
    num_objs
        Number of objects (cells) to sample from each mpp_data.

    Returns
    -------
    dictionary containing clustered cells in `cluster_name` and coloured cells in `cluster_name_colored`.
    """
    from campa.pl import annotate_img

    cl.set_cluster_name(cluster_name)
    res: dict[str, dict[str, Any]] = {cluster_name: {}, cluster_name + "_colored": {}}
    for data_dir, mpp_data in mpp_datas.items():
        print(data_dir)
        # get random obj_ids for this mpp_data
        rng = np.random.default_rng(seed=42)
        obj_ids = rng.choice(mpp_data.unique_obj_ids, num_objs, replace=False)
        # subset mpp_data to obj_ids
        sub_mpp_data = mpp_data.subset(obj_ids=obj_ids, copy=True)
        # project_clustering
        sub_mpp_data = cl.project_clustering(sub_mpp_data)

        # if only need colored cells, can pass annotation_kwargs to get_object_imgs
        res[cluster_name][data_dir] = sub_mpp_data.get_object_imgs(
            data=cluster_name
        )  # annotation_kwargs={'color': True, 'annotation': cl.cluster_annotation})
        res[cluster_name + "_colored"][data_dir] = [
            annotate_img(img, annotation=cl.cluster_annotation, from_col=cluster_name, color=True)
            for img in res[cluster_name][data_dir]
        ]
    return res
