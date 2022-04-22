from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Mapping, Iterable, Optional
from functools import partial
import os
import time
import logging
import multiprocessing

from numba import jit
from skimage.measure import label, regionprops
import numpy as np
import pandas as pd
import anndata as ad
import squidpy as sq
import numba.types as nt

from campa.data import MPPData
from campa.constants import CoOccAlgo, campa_config
from campa.tl._cluster import annotate_clustering
from campa.tl._experiment import Experiment

ft = nt.float32
it = nt.int64

ft = nt.float32
it = nt.int64


def thresholded_count(df, threshold=0.9):
    """
    Count largest CSL objects per cell.

    Sort objects by area (largest areas first) and count how many are needed to exceed threshold % of the total area.
    This is essentially a small object invariant way of counting big objects.
    Can be used as aggregation function in :meth:`FeatureExtractor.get_object_stats`.

    Parameters
    ----------
    threshold
        Only consider large objects up to cumsum of 90% of the total area.

    Returns
    -------
    count
    """
    total = df.sum()
    cumsum = (df / total).sort_values(ascending=False).cumsum()
    count = (cumsum < threshold).sum() + 1
    return count


def thresholded_median(df, threshold=0.9):
    """
    Calculate median area of large CSL objects per cell.

    Sort objects by area (largest areas first)
    and compute the median area of all objects that are below cumsum of threshold.
    This is essentially a small object invariant way of computing median.
    Can be used as aggregation function in :meth:`FeatureExtractor.get_object_stats`.

    Parameters
    ----------
    threshold
        Only consider large objects up to cumsum of 90% of the total area.

    Returns
    -------
    median
    """
    total = df.sum()
    cumsum = (df / total).sort_values(ascending=False).cumsum()
    mask = cumsum < threshold
    # object that will finally exceed threshold (would like to include in median calc, as is also included in count)
    idx = (cumsum >= threshold).idxmax()
    mask[idx] = True
    median = df[cumsum[mask].index].median()
    return median


def extract_features(params: Mapping[str, Any]) -> None:
    """
    Extract features from clustered dataset using :class:`FeaturesExtractor`.

    Creates features :class:`anndata.AnnData` object.

    Params determine what features are extracted from a given clustering.
    The following keys in params are expected:

    - ``experiment_dir``: path to experiment directory relative to campa_config.EXPERIMENT_DIR.
    - ``cluster_name``: name of clustering to use.
    - ``cluster_dir``: dir of subsampled clustering to load annotation.
      Relative to experiment_dir.
      Default is taking first of ``experiment_dir/aggregated/sub-*``.
    - ``cluster_col``: cluster annotation to use. Defaults to ``cluster_name``.
    - ``data_dirs``: data dirs to be processed.
      Relative to ``experiment_dir/aggregated/full_data``.
      If None, all available data_dirs will be processed.
    - ``save_name``: filename to use for saving extracted features.
    - ``force``: force calculation even when adata exists.
    - ``features``: type of features to extract. One or more of `intensity`, `co-occurrence`, `object-stats`.

        - Intensity: per-cluster mean and size features. Needs to be calculated first to set up the adata.
        - Co-occurrence: spatial co-occurrence between pairs of clusters at different distances.
        - Object stats: number and area of connected components per cluster.

    - ``co_occurrence_params``: parameters for co-occurrence calculation.

        - ``min``, ``max``, ``nsteps``: size of distances interval.
        - ``logspace``: use log spacing of co-occurrence intervals.
        - ``num_processes``:  number of processes to use to compute co-occurrence scores.

    - ``object_stats_params``: parameter dict for object-stats calculation.

        - ``features``: features to extract in mode object-stats.
          Possible features: `area`, `circularity`, `elongation`, `extent`.
        - ``channels``: intensity channels to extract mean per cluster from.

    Parameters
    ----------
    params
        Parameter dictionary.
    """
    # set up FeatureExtractor
    log = logging.getLogger("extract_features")
    exp = Experiment.from_dir(params["experiment_dir"])
    data_dirs = params["data_dirs"]
    if data_dirs is None or len(data_dirs) == 0:
        data_dirs = exp.data_params["data_dirs"]

    for data_dir in data_dirs:
        log.info(f'extracting features {params["features"]} from {data_dir}')
        adata_fname = os.path.join(exp.full_path, "aggregated/full_data", data_dir, params["save_name"])
        if os.path.exists(adata_fname):
            log.info(f"initialising from existing adata {adata_fname}")
            extr = FeatureExtractor.from_adata(adata_fname)
        else:
            extr = FeatureExtractor(
                exp,
                data_dir=data_dir,
                cluster_name=params["cluster_name"],
                cluster_dir=params["cluster_dir"],
                cluster_col=params["cluster_col"],
            )
        # extract features
        if "intensity" in params["features"]:
            extr.extract_intensity_size(force=params["force"], fname=params["save_name"])
        if "co-occurrence" in params["features"]:
            co_occ_params = params["co_occurrence_params"]
            if co_occ_params["logspace"]:
                interval = np.logspace(
                    np.log2(co_occ_params["min"]),
                    np.log2(co_occ_params["max"]),
                    co_occ_params["nsteps"],
                    base=2,
                ).astype(np.float32)
            else:
                interval = np.linspace(co_occ_params["min"], co_occ_params["max"], co_occ_params["nsteps"]).astype(
                    np.float32
                )
            extr.extract_co_occurrence(interval=interval, num_processes=co_occ_params["num_processes"])
        if "object-stats" in params["features"]:
            obj_params = params["object_stats_params"]
            extr.extract_object_stats(
                features=obj_params["features"],
                intensity_channels=obj_params["channels"],
            )


class FeatureExtractor:
    """
    Extract features from clustering.

    Parameters
    ----------
    exp
        Experiment to extract features from.
    data_dir
        Name of data to cluster. Relative to ``{exp.full_path}/aggregated/full_data``.
    cluster_name
        Name of clustering to use.
    cluster_dir
        Dir of subsampled clustering to load annotation.
        Relative to ``exp.full_path```.
        Default is taking first of ``{exp.full_path}/aggregated/sub-*``.
    cluster_col
        Cluster annotation to use. Defaults to ``cluster_name``.
    adata
        If existing, the features adata object containing extracted features.
    """

    def __init__(
        self,
        exp: Experiment,
        data_dir: str,
        cluster_name: str,
        cluster_dir: Optional[str] = None,
        cluster_col: Optional[str] = None,
        adata: Optional[ad.AnnData] = None,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        cluster_col = cluster_col if cluster_col is not None else cluster_name
        self.params = {
            "data_dir": data_dir,
            "cluster_name": cluster_name,
            "cluster_dir": cluster_dir,
            "cluster_col": cluster_col,
            "exp_name": exp.dir + "/" + exp.name,
        }
        self.adata = adata

        self.annotation = self.exp.get_cluster_annotation(str(self.params["cluster_name"]), self.params["cluster_dir"])
        clusters = list(np.unique(self.annotation[self.params["cluster_col"]]))
        clusters.remove("")
        self.clusters = clusters

        self._mpp_data: Union[MPPData, None] = None

    @classmethod
    def from_adata(cls, fname: str) -> "FeatureExtractor":
        """
        Initialise from existing features :class:`ad.AnnData` object.

        Parameters
        ----------
        fname
            Full path to adata object.
        """
        adata = ad.read(fname)
        params = deepcopy(adata.uns["params"])
        exp = Experiment.from_dir(params.pop("exp_name"))
        self = cls(exp, adata=adata, **params)
        self.fname = fname
        return self

    @property
    def mpp_data(self) -> MPPData:
        """
        :class:`MPPData` object containing pixel-wise clustered data from ``data_dir``.
        """
        if self._mpp_data is None:
            self._mpp_data = MPPData.from_data_dir(
                str(self.params["data_dir"]),
                base_dir=os.path.join(self.exp.full_path, "aggregated/full_data"),
                keys=["x", "y", "mpp", "obj_ids", str(self.params["cluster_name"])],
                data_config=self.exp.config["data"]["data_config"],
            )
            # ensure that cluster data is string
            self._mpp_data._data[str(self.params["cluster_name"])] = self._mpp_data._data[
                str(self.params["cluster_name"])
            ].astype(str)
            # prepare according to data_params
            data_params = deepcopy(self.exp.data_params)
            self._mpp_data.prepare(data_params)
        return self._mpp_data

    def extract_intensity_size(self, force: bool = False, fname: str = "features.h5ad") -> None:
        """
        Calculate per cluster mean intensity and size for each object.

        Saves adata in ``{exp.full_path}/aggregated/full_data/data_dir/{fname}``

        Parameters
        ----------
        force
            Overwrite existing adata.
        fname
            Name of the saved adata.
        """
        if self.adata is not None and not force:
            self.log.info("extract_intensity_size: adata is not None. Specify force=True to overwrite. Exiting.")
            return
        self.log.info(
            f"Calculating {self.params['cluster_name']} (col: {self.params['cluster_col']})"
            + f" mean and size for {self.params['data_dir']}"
        )
        df = pd.DataFrame(
            data=self.mpp_data.center_mpp,
            columns=list(self.mpp_data.channels.name),
            index=self.mpp_data.obj_ids,
        )
        # create adata with X = mean intensity of "all" cluster
        grouped = df.groupby(df.index)
        adata = ad.AnnData(X=grouped.mean())

        # add all metadata
        OBJ_ID = self.mpp_data.data_config.OBJ_ID
        metadata = self.mpp_data.metadata.copy()
        metadata[OBJ_ID] = metadata[OBJ_ID].astype(str)
        metadata = pd.merge(metadata, adata.obs, how="right", left_on=OBJ_ID, right_index=True)
        metadata = metadata.reset_index(drop=True)  # make new index, keep mapobject_id in column
        metadata.index = metadata.index.astype(str)  # make index str, because adata does not play nice with int indices
        # add int col of mapobject_id for easier merging
        metadata["obj_id_int"] = metadata[OBJ_ID].astype(int)
        adata.obs = metadata

        # add size of all cluster
        # reindex to ensure all object are present
        size = grouped[list(df.columns)[0]].count().reindex(adata.obs["obj_id_int"])
        adata.obsm["size"] = pd.DataFrame(columns=["all"] + self.clusters, index=adata.obs.index)
        adata.obsm["size"]["all"] = np.array(size)

        # add uns metadata
        adata.uns["clusters"] = self.clusters
        adata.uns["params"] = self.params

        # add intensities of each cluster as a layer in adata and fill size obsm
        for c in self.clusters:
            self.log.debug(f"processing {c}")
            # get cluster ids to mask
            c_ids = list(self.annotation[self.annotation[self.params["cluster_col"]] == c][self.params["cluster_name"]])
            mask = np.where(np.isin(self.mpp_data.data(str(self.params["cluster_name"])), c_ids))
            cur_df = df.iloc[mask]
            # group by obj_id
            grouped = cur_df.groupby(cur_df.index)
            # add mean of cluster
            # reindex to ensure all object are present
            mean = grouped.mean().reindex(adata.obs["obj_id_int"])
            mean = mean.fillna(0)
            adata.layers[f"intensity_{c}"] = np.array(mean[adata.var.index])
            # add size
            # reindex to ensure all object are present
            size = grouped[list(df.columns)[0]].count().reindex(adata.obs["obj_id_int"])
            adata.obsm["size"][c] = np.array(size)
        # fill nans in size obsm
        adata.obsm["size"] = adata.obsm["size"].fillna(0)

        self.adata = adata

        # write to disk
        fname = os.path.join(self.exp.full_path, "aggregated/full_data", str(self.params["data_dir"]), fname)
        self.log.info(f"saving adata to {fname}")
        self.fname = fname
        self.adata.write(self.fname)

    def extract_object_stats(
        self,
        features: Iterable[str] = ("area", "circularity", "elongation", "extent"),
        intensity_channels: Iterable[str] = (),
    ) -> None:
        """
        Extract features from connected components per cluster for each cell.

        Implemented features are: `area`, `circlularity`, `elongation`, and `extent` of connected components
        per cluster for each cell.
        In addition, the mean intensity per component/region of
        channels specified in intensity_channels is calculated.
        Per component/region features are calculated and stored in ``uns['object_stats']``,
        together with OBJ_ID and cluster that this region belongs to.
        To aggregate these computed stats in a mean/median values per OBJ_ID,
        use :meth:`FeatureExtractor.get_object_stats`.

        $circularity = (4 * pi * Area) / Perimeter^2$

        $elongation = (major_axis - minor_axis) / major_axis$

        Parameters
        ----------
        features
            List of features to be calculated.
        intensity_channels
            List of channels for which the mean intensity should be extracted.

        Returns
        -------
        Nothing, modifies ``adata``:
        - Adds ``uns`` entry: ``object_stats`` and ``object_stats_params``
        - Adds ``obs`` entries: ``mean_{channel_name}``
        """
        if self.adata is None:
            self.log.info(
                "extract_object_stats: adata is None."
                + " Calculate it with extract_intensity_size before extracting object stats. Exiting."
            )
            return
        if features is None:
            features = []
        features = list(features)
        if intensity_channels is None:
            intensity_channels = []
        intensity_channels = list(intensity_channels)
        assert len(features) + len(intensity_channels) > 0, "nothing to compute"
        self.log.info(
            f"calculating object stats {features} and channels {intensity_channels}"
            + f" for clustering {self.params['cluster_name']} (col: {self.params['cluster_col']})"
        )
        cluster_names = {n: i for i, n in enumerate(self.clusters + [""])}

        feature_names = features
        intensity_feature_names = [f"mean_{ch}" for ch in intensity_channels]
        features: Dict[str, List[Any]] = {
            feature: [] for feature in feature_names + intensity_feature_names + ["mapobject_id", "clustering"]
        }  # ignore: type[assignment]
        for obj_id in self.mpp_data.unique_obj_ids:
            mpp_data = self.mpp_data.subset(obj_ids=[obj_id], copy=True)
            img, _ = mpp_data.get_object_img(
                obj_id,
                data=str(self.params["cluster_name"]),
                annotation_kwargs={
                    "annotation": self.annotation,
                    "to_col": self.params["cluster_col"],
                },
            )
            # convert labels to numbers
            img = np.vectorize(cluster_names.__getitem__)(img[:, :, 0])
            label_img = label(img, background=len(self.clusters), connectivity=2)
            # define intensity image
            intensity_img = img[:, :, np.newaxis]
            if len(intensity_channels) > 0:
                obj_img, _ = mpp_data.get_object_img(
                    obj_id,
                    data="mpp",
                    channel_ids=mpp_data.get_channel_ids(intensity_channels),
                )
                intensity_img = np.concatenate([intensity_img, obj_img], axis=-1)
            # iterate over all regions in this image
            for region in regionprops(label_img, intensity_image=intensity_img):
                # filter out single pixel regions, they cause problems in elongation and circularity calculation
                if region.area > 1:
                    # get cluster label for this region
                    assert region.min_intensity[0] == region.max_intensity[0]
                    c = int(region.min_intensity[0])
                    # add clustering and obj_id
                    features["clustering"].append((self.clusters + [""])[c])
                    features["mapobject_id"].append(obj_id)
                    # add all other features
                    for feature in feature_names:
                        if feature == "area":
                            features[feature].append(region.area)
                        elif feature == "circularity":
                            # circularity can max be 1,
                            # larger values are due to tiny regions where perimeter is overestimated
                            features[feature].append(min(4 * np.pi * region.area / region.perimeter ** 2, 1))
                        elif feature == "elongation":
                            features[feature].append(
                                (region.major_axis_length - region.minor_axis_length) / region.major_axis_length
                            )
                        elif feature == "extent":
                            features[feature].append(region.extent)
                        else:
                            raise NotImplementedError(feature)
                    # get intensity features for this region
                    for i, _ in enumerate(intensity_channels):
                        # mean intensity is in channel i+1 of intensity_img (channel 0 is cluster image)
                        features[intensity_feature_names[i]].append(region.mean_intensity[i + 1])

        df = pd.DataFrame(features)
        self.adata.uns["object_stats"] = df
        self.adata.uns["object_stats_params"] = {
            "features": feature_names,
            "intensity_channels": intensity_channels,
        }

        # write adata
        self.log.info(f"saving adata to {self.fname}")
        self.log.info(f'adata params {self.adata.uns["params"]}')
        self.adata.write(self.fname)

    def extract_co_occurrence(
        self,
        interval: Iterable[float],
        algorithm: Union[str, CoOccAlgo] = CoOccAlgo.OPT,
        num_processes: Optional[int] = None,
    ) -> None:
        """
        Extract co_occurrence for each cell invididually.

        TODO: add reset flag, that sets existing co-occ matrices to 0 before running co_occ algo.

        Parameters
        ----------
        interval
            Distance intervals for which to calculate co-occurrence score.
        algorithm
            Co-occurrence function to use:
            - `squidpy`: use :func:`sq.gr.co_occurrence`.
            -  `opt`: use custom implementation which is optimised for a large number of pixels.
                This implementation avoids recalculation of distances, using the fact that coordinates
                in given images lie on a regular grid.
                Use opt for very large inputs.

        num_processes
            only for ``algorithm='opt'``. Number if processes to use to compute scores.

        Returns
        -------
        Nothing, modifies ``adata``
        - Adds ``obsm`` entries: ``co_occurrence_{cluster1}_{cluster2}``
        """
        if self.adata is None:
            self.log.info(
                "extract_co_occurrence: adata is None."
                + " Calculate it with extract_intensity_size before extracting co_occurrence. Exiting."
            )
            return
        self.log.info(
            f"calculating co-occurrence for intervals {interval} and clustering {self.params['cluster_name']}"
            + f" (col: {self.params['cluster_col']})"
        )
        if CoOccAlgo(algorithm) == CoOccAlgo.OPT:
            cluster_names = {n: i for i, n in enumerate(self.clusters + [""])}
            coords2_list = _prepare_co_occ(interval)
        elif CoOccAlgo(algorithm) == CoOccAlgo.SQUIDPY:
            cluster_names = {n: i for i, n in enumerate(self.clusters)}

        obj_ids = []
        co_occs = []
        chunks = 20
        i = 0
        missing_obj_ids = self._missing_co_occ_obj_ids()
        self.log.info(f"calculating co-occurrence for {len(missing_obj_ids)} objects")
        for obj_id in missing_obj_ids:
            if CoOccAlgo(algorithm) == CoOccAlgo.OPT:
                mpp_data = self.mpp_data.subset(obj_ids=[obj_id], copy=True)
                img, (pad_x, pad_y) = mpp_data.get_object_img(
                    obj_id,
                    data=str(self.params["cluster_name"]),
                    annotation_kwargs={
                        "annotation": self.annotation,
                        "to_col": self.params["cluster_col"],
                    },
                )
                # convert labels to numbers
                img = np.vectorize(cluster_names.__getitem__)(img)
                clusters1 = np.vectorize(cluster_names.__getitem__)(
                    annotate_clustering(
                        mpp_data._data[str(self.params["cluster_name"])],
                        self.annotation,
                        str(self.params["cluster_name"]),
                        self.params["cluster_col"],
                    )
                )
                # shift coords according to image padding, st coords correspond to img coords
                coords1 = (np.array([mpp_data.x, mpp_data.y]) - np.array([pad_x, pad_y])[:, np.newaxis]).astype(
                    np.int64
                )
                self.log.info(f"co-occurrence for {obj_id}, with {len(mpp_data.x)} elements")
                co_occ = _co_occ_opt(
                    coords1,
                    coords2_list,
                    clusters1,
                    img,
                    num_clusters=len(self.clusters),
                    num_processes=num_processes,
                )
            elif CoOccAlgo(algorithm) == CoOccAlgo.SQUIDPY:
                adata = self.mpp_data.subset(obj_ids=[obj_id], copy=True).get_adata(
                    obs=[str(self.params["cluster_name"])]
                )
                # ensure that cluster annotation is present in adata
                if self.params["cluster_name"] != self.params["cluster_col"]:
                    adata.obs[self.params["cluster_col"]] = annotate_clustering(
                        adata.obs[self.params["cluster_name"]],
                        self.annotation,
                        str(self.params["cluster_name"]),
                        self.params["cluster_col"],
                    )
                adata.obs[str(self.params["cluster_col"])] = adata.obs[self.params["cluster_col"]].astype("category")
                self.log.info(f"co-occurrence for {obj_id}, with shape {adata.shape}")
                cur_co_occ, _ = sq.gr.co_occurrence(
                    adata,
                    cluster_key=self.params["cluster_col"],
                    spatial_key="spatial",
                    interval=interval,
                    copy=True,
                    show_progress_bar=False,
                    n_splits=1,
                )
                # ensure that co_occ has correct format incase of missing clusters
                co_occ = np.zeros((len(self.clusters), len(self.clusters), len(list(interval)) - 1))
                cur_clusters = np.vectorize(cluster_names.__getitem__)(
                    np.array(adata.obs[self.params["cluster_col"]].cat.categories)
                )
                grid = np.meshgrid(cur_clusters, cur_clusters)
                co_occ[grid[0].flat, grid[1].flat] = cur_co_occ.reshape(-1, len(list(interval)) - 1)
            co_occs.append(co_occ.copy())
            obj_ids.append(obj_id)

            i += 1
            if (i % chunks == 0) or (obj_id == missing_obj_ids[-1]):
                # save
                self.log.info(f"Saving chunk {i-chunks}-{i}")
                # add info to adata
                co_occ = np.array(co_occs)
                for i1, c1 in enumerate(self.clusters):
                    for i2, c2 in enumerate(self.clusters):
                        df = pd.DataFrame(
                            co_occ[:, i1, i2],
                            index=obj_ids,
                            columns=np.arange(len(list(interval)) - 1).astype(str),
                        )
                        df.index = df.index.astype(str)
                        # ensure obj_ids are in correct order
                        df = pd.merge(
                            df,
                            self.adata.obs,
                            how="right",
                            left_index=True,
                            right_on="mapobject_id",
                            suffixes=("", "right"),
                        )[df.columns]
                        df = df.fillna(0)
                        # add to adata.obsm
                        if f"co_occurrence_{c1}_{c2}" in self.adata.obsm:
                            self.adata.obsm[f"co_occurrence_{c1}_{c2}"] += df
                        else:
                            self.adata.obsm[f"co_occurrence_{c1}_{c2}"] = df
                self.adata.uns["co_occurrence_params"] = {"interval": list(interval)}
                self.log.info(f"saving adata to {self.fname}")
                self.log.info(f'adata params {self.adata.uns["params"]}')
                self.adata.write(self.fname)
                # reset co_occ list and obj_ids
                co_occs = []
                obj_ids = []

    def get_intensity_adata(self) -> ad.AnnData:
        """
        Adata object with intensity per cluster combined in X.

        Needed for intensity and dotplots.
        """
        if self.adata is None:
            raise ValueError("adata is None, need to calculate first.")
        adata = self.adata
        adatas = {}
        cur_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
        cur_adata.obs["size"] = adata.obsm["size"]["all"]
        adatas["all"] = cur_adata
        for c in adata.uns["clusters"]:
            cur_adata = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var)
            cur_adata.X = adata.layers[f"intensity_{c}"]
            cur_adata.obs["size"] = adata.obsm["size"][c]
            adatas[c] = cur_adata
        comb_adata = ad.concat(adatas, uns_merge="same", index_unique="-", label="cluster")
        return comb_adata

    def get_object_stats(
        self, area_threshold: int = 10, agg: Union[Iterable[str], Mapping[str, str]] = ("median",), save: bool = False
    ) -> pd.DataFrame:
        """
        Aggregate object stats per obj_id.

        Parameters
        ----------
        area_threshold
            All components smaller than this threshold are discarded.
        agg
            List of aggregation function or dict with ``{feature: function}``.
            Passed to :func:`pd.GroupBy.agg`
        save
            Save adata object with `object_stats_agg` entry in ``obsm``.

        Returns
        -------
        pd.DataFrame
            obj_id x (clustering, feature) dataframe

        Additionally stores result in ``self.adata.obsm['object_stats_agg']``
        """
        if self.adata is None:
            raise ValueError("adata is None, need to calculate first.")
        assert (
            "object_stats" in self.adata.uns.keys()
        ), "No object stats found. Use self.extract_object_stats to calculate."
        OBJ_ID = campa_config.get_data_config(self.exp.config["data"]["data_config"]).OBJ_ID
        df = self.adata.uns["object_stats"]
        # filter out small regions
        df = df[df["area"] > area_threshold]
        grp = df.groupby(["clustering", OBJ_ID])
        agg_stats = grp.agg(agg)
        # rename columns to feature_agg
        agg_stats.columns = [f"{i}_{j}" for i, j in agg_stats.columns]
        # add count column
        agg_stats["count"] = grp.count()["area"]
        # reshape to obj_id x (clustering, feature)
        agg_stats = agg_stats.unstack(level=0)
        # replace nan with 0 (clusters that do not exist in given obj)
        agg_stats = agg_stats.fillna(0)

        # ensure obj_ids are in correct order
        agg_stats.index = agg_stats.index.astype(str)
        agg_stats.columns = [(i, j) for i, j in agg_stats.columns]
        agg_stats = pd.merge(
            agg_stats,
            self.adata.obs,
            how="right",
            left_index=True,
            right_on=OBJ_ID,
            suffixes=("", "right"),
        )[agg_stats.columns]
        agg_stats = agg_stats.fillna(0)
        agg_stats.columns = pd.MultiIndex.from_tuples(agg_stats.columns)
        # store result in adata
        self.adata.obsm["object_stats_agg"] = deepcopy(agg_stats)
        # flatten columns to allow saving adata
        self.adata.obsm["object_stats_agg"].columns = [
            f"{i}|{j}" for i, j in self.adata.obsm["object_stats_agg"].columns
        ]
        if save:
            self.adata.write(self.fname)
        return agg_stats

    def extract_intensity_csv(self, obs: Optional[Iterable[str]] = None) -> None:
        """
        Extract csv file containing obj_id, mean cluster intensity and size for each channel.

        Saves csv as ``export/intensity_{self.fname}.csv``.

        Parameters
        ----------
        obs
            column names from `metadata.csv` that should be additionally stored.
        """
        if self.adata is None:
            self.log.warning(
                "Intensity and size information is not present. Calculate extract_intensity_size first! Exiting."
            )
            return
        adata = self.get_intensity_adata()
        df = pd.DataFrame(data=adata.X, columns=adata.var_names)
        # add size
        df["size"] = np.array(adata.obs["size"])
        # add cluster and obj_id
        OBJ_ID = campa_config.get_data_config(self.exp.config["data"]["data_config"]).OBJ_ID
        df["cluster"] = np.array(adata.obs["cluster"])
        df[OBJ_ID] = np.array(adata.obs[OBJ_ID])
        # add additional obs
        if obs is not None:
            for col in obs:
                df[col] = np.array(adata.obs[col])
        # save csv
        dirname = os.path.join(os.path.dirname(self.fname), "export")
        os.makedirs(dirname, exist_ok=True)
        df.to_csv(os.path.join(dirname, f"intensity_{os.path.basename(os.path.splitext(self.fname)[0])}.csv"))

    def extract_object_stats_csv(
        self, obs: Optional[Iterable[str]] = None, features: Iterable[str] = None, clusters: Iterable[str] = None
    ) -> None:
        """
        Extract csv files containing obj_id, co_occurrence scores at each distance
        interval for every cluster-cluster pair.

        Saves csv as ``export/co_occurrence_{cluster1}_{cluster2}_{self.fname}.csv``.

        Parameters
        ----------
        obs
            column names from `metadata.csv` that should be additionally stored.
        clusters
            List of cluster names for which pairwise co_occurrence scores should be calculated
        features: list of features to display. Must be columns of adata.obsm['object_stats_agg'].
                If None, all features are displayed.
            clusters: list of clusters to display. Must be columns of adata.obsm['object_stats_agg'].
                If None, all clusters are displayed.
        """
        if self.adata is None or "object_stats_agg" not in self.adata.obsm:
            self.log.warn(
                "Object stats information is not present.\
                Run extract_object_stats and get_objects_stats first! Exiting."
            )
            return

        agg_stats = deepcopy(self.adata.obsm["object_stats_agg"])
        if not isinstance(agg_stats.columns, pd.MultiIndex):
            # restore multiindex for easier access
            agg_stats.columns = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in agg_stats.columns])
        if features is None:
            features = agg_stats.columns.levels[0]
        if clusters is None:
            clusters = agg_stats.columns.levels[1]

        df = {}
        for feature in features:
            for cluster in clusters:
                name = f"{feature}|{cluster}"
                df[name] = agg_stats[feature][cluster]
        df = pd.DataFrame(df)
        # add obj_id
        OBJ_ID = campa_config.get_data_config(self.exp.config["data"]["data_config"]).OBJ_ID
        df[OBJ_ID] = np.array(self.adata.obs[OBJ_ID])
        # add additional obs
        if obs is not None:
            for col in obs:
                df[col] = np.array(self.adata.obs[col])
        # save csv
        dirname = os.path.join(os.path.dirname(self.fname), "export")
        os.makedirs(dirname, exist_ok=True)
        df.to_csv(os.path.join(dirname, f"object_stats_{os.path.basename(os.path.splitext(self.fname)[0])}.csv"))

    def extract_co_occurrence_csv(
        self, obs: Optional[Iterable[str]] = None, clusters: Optional[Iterable[str]] = None
    ) -> None:
        """
        Extract csv files containing obj_id,
        co_occurrence scores at each distance interval for every cluster-cluster pair.

        Saves csv as ``export/co_occurrence_{cluster1}_{cluster2}_{self.fname}.csv``.

        Parameters
        ----------
        obs
            column names from `metadata.csv` that should be additionally stored.
        clusters
            List of cluster names for which pairwise co_occurrence scores should be calculated
        """
        if self.adata is None:
            self.log.warn("Co-occurrence information is not present. Calculate extract_co_occurrence first! Exiting.")
            return
        if clusters is None:
            clusters = self.adata.uns["clusters"]
        for c1 in clusters:
            for c2 in clusters:
                columns = list(
                    map(
                        lambda x: f"{x[0]:.2f}-{x[1]:.2f}",
                        zip(
                            self.adata.uns["co_occurrence_params"]["interval"][:-1],
                            self.adata.uns["co_occurrence_params"]["interval"][1:],
                        ),
                    )
                )
                df = self.adata.obsm[f"co_occurrence_{c1}_{c2}"].copy()
                df.columns = columns
                # add obj_id
                OBJ_ID = campa_config.get_data_config(self.exp.config["data"]["data_config"]).OBJ_ID
                df[OBJ_ID] = np.array(self.adata.obs[OBJ_ID])
                # add additional obs
                if obs is not None:
                    for col in obs:
                        df[col] = np.array(self.adata.obs[col])
                # save csv
                dirname = os.path.join(os.path.dirname(self.fname), "export")
                os.makedirs(dirname, exist_ok=True)
                df.to_csv(
                    os.path.join(
                        dirname, f"co_occurrence_{c1}_{c2}_{os.path.basename(os.path.splitext(self.fname)[0])}.csv"
                    )
                )

    def _missing_co_occ_obj_ids(self):
        """
        Return those obj_ids that do not yet have co-occurence scores calculated.
        """
        if self.adata is None:
            raise ValueError("adata is None, need to calculate first.")
        n = f"co_occurrence_{self.clusters[0]}_{self.clusters[0]}"
        if n not in self.adata.obsm.keys():
            # no co-occ calculated
            return self.mpp_data.unique_obj_ids
        else:
            OBJ_ID = campa_config.get_data_config(self.exp.config["data"]["data_config"]).OBJ_ID
            masks = []
            for c1 in self.clusters:
                for c2 in self.clusters:
                    self.adata.obsm[f"co_occurrence_{c1}_{c2}"]
                    masks.append((self.adata.obsm[f"co_occurrence_{c1}_{c2}"] == 0).all(axis=1))
            obj_ids = np.array(self.adata[np.array(masks).T.all(axis=1)].obs[OBJ_ID]).astype(np.uint32)
            return obj_ids

    def compare(self, obj: "FeatureExtractor") -> Tuple[bool, Dict[str, Any]]:
        """
        Compare feature extractors.

        Compares all features contained in adata and annotation dict.

        Parameters
        ----------
        obj
            Object to compare to.

        Returns
        -------
        Tuple of (``overall_result``, ``results_dict``).
            ``overall_result`` is True, if all data in ``results_dict`` is True.
            ``results_dict`` contains for each tested key True or False.
        """

        def array_comp(arr1, arr2):
            comp = arr1 == arr2
            if comp is False:
                return False
            else:
                return comp.all()

        if self.adata is None or obj.adata is None:
            self.log.warning("Cannot compare FeatureExtractors, one or more adatas is None")
            return False, {}
        results_dict = {}
        results_dict["annotation"] = self.annotation.equals(obj.annotation)
        res = {}
        res["X"] = array_comp(self.adata.X, obj.adata.X)
        res["obs"] = self.adata.obs.equals(obj.adata.obs)
        res["obsm"] = {}
        for key in self.adata.obsm.keys():
            res["obsm"][key] = self.adata.obsm[key].equals(obj.adata.obsm[key])
        res["layers"] = {}
        for key in self.adata.layers.keys():
            res["layers"][key] = array_comp(self.adata.layers[key], obj.adata.layers[key])
        res["uns"] = {}
        for key in self.adata.uns.keys():
            if key == "params":
                # params are experiment specific, but here we just care about the resulting values
                continue
            if key == "object_stats":
                res["uns"][key] = self.adata.uns[key].equals(obj.adata.uns[key])
            elif key == "clusters":
                res["uns"][key] = array_comp(self.adata.uns[key], obj.adata.uns[key])
            elif key in ("object_stats_params", "co_occurrence_params"):
                res["uns"][key] = {}
                for k in self.adata.uns[key].keys():
                    res["uns"][key][k] = array_comp(self.adata.uns[key][k], obj.adata.uns[key][k])
            else:
                res["uns"][key] = self.adata.uns[key] == obj.adata.uns[key]
        results_dict["adata"] = res

        # summarise results
        # flatten dict
        df = pd.json_normalize(results_dict, sep="_")
        vals = df.to_dict(orient="records")[0].values()
        return all(list(vals)), results_dict


@jit(ft[:, :](it[:], it[:], it), fastmath=True)
def _count_co_occ(clus1: np.ndarray, clus2: np.ndarray, num_clusters: int) -> np.ndarray:
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
    cur_coords = cur_coords.reshape((2, -1))
    clus1 = clus1.reshape([-1])

    # filter cur_coords that are outside image
    shape = np.expand_dims(np.array([img.shape[1], img.shape[0]]), 1)
    mask = np.all((cur_coords >= 0) & (cur_coords < shape), axis=0)
    cur_coords = cur_coords[:, mask]
    clus1 = clus1[mask]

    # get cluster of cur_coords
    clus2 = img[cur_coords[1], cur_coords[0]].flatten()

    # remove those pairs where clus2 is outside of this image (cluster id is not a valid id)
    mask = clus2 < num_clusters
    # assert (clus1 < num_clusters).all()
    clus1 = clus1[mask]
    clus2 = clus2[mask]

    co_occur = _count_co_occ(clus1, clus2, num_clusters)
    return co_occur


def _co_occ_opt(
    coords1: np.ndarray,  # int64
    coords2_list: np.ndarray,  # int64
    clusters1: np.ndarray,  # int64
    img: np.ndarray,  # int64
    num_clusters: int,
    num_processes: Optional[int] = None,
) -> np.ndarray:
    """
    Calculate co-occurrence scores for several intervals.

    For decreased memory usage coords1 x coords2 pairs are chunked in CO_OCC_CHUNK_SIZE chunks and processed.
    If num_processes is specified, uses multiprocessing to calculate co-occurrence scores.

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
    log = logging.getLogger("_co_occ_opt")
    if num_processes is not None:
        pool = multiprocessing.Pool(num_processes)
        log.info(f"using {num_processes} processes to calculate co-occ scores.")
    if coords1.shape[1] > campa_config.CO_OCC_CHUNK_SIZE:
        raise ValueError(
            f"coords1 with size {coords1.shape[1]} is larger than CO_OCC_CHUNK_SIZE {campa_config.CO_OCC_CHUNK_SIZE}."
            + " Cannot compute _co_occ_opt"
        )
    out = np.zeros((num_clusters, num_clusters, len(coords2_list)), dtype=np.float32)
    # iterate over each interval
    for idx, coords2 in enumerate(coords2_list):
        # log.info(f'co occ for interval {idx+1}/{len(coords2_list)},
        # with {coords1.shape[1]} x {coords2.shape[1]} coord pairs')
        if (coords1.shape[1] * coords2.shape[1]) > campa_config.CO_OCC_CHUNK_SIZE:
            chunk_size = int(campa_config.CO_OCC_CHUNK_SIZE / coords1.shape[1])
            coords2_chunks = np.split(coords2, np.arange(0, coords2.shape[1], chunk_size), axis=1)
            # log.info(f'splitting coords2 in {len(coords2_chunks)} chunks')
        else:
            coords2_chunks = [coords2]

        # calculate pairwise cluster counts
        time.time()
        co_occur = np.zeros((num_clusters, num_clusters), dtype=np.float32)
        map_fn = partial(
            _co_occ_opt_helper,
            coords1=coords1,
            clusters1=clusters1,
            img=img,
            num_clusters=num_clusters,
        )
        if num_processes is not None:
            # for res in tqdm.tqdm(pool.imap_unordered(map_fn, coords2_chunks), total=len(coords2_chunks)):
            for res in pool.imap_unordered(map_fn, coords2_chunks):
                co_occur += res
        else:
            for res in map(map_fn, coords2_chunks):
                co_occur += res

        time.time()
        # log.info(f'calculating co_occur for these coords took {t2-t1:.0f}s.')

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
    Return lists of coordinates to consider for each interval. Coordinates are relative to [0,0].
    """
    arr = np.zeros((int(interval[-1]) * 2 + 1, int(interval[-1]) * 2 + 1))
    # calc distances for interval range (assuming c as center px)
    c = int(interval[-1]) + 1
    c: np.ndarray = np.array([c, c]).T
    xx, yy = np.meshgrid(np.arange(len(arr)), np.arange(len(arr)))
    coords = np.array([xx.flatten(), yy.flatten()]).T
    dists = np.sqrt(np.sum((coords - c) ** 2, axis=-1))
    dists = dists.reshape(int(interval[-1]) * 2 + 1, -1)
    # calc coords for each thresh interval
    coord_lists = []
    for thres_min, thres_max in zip(interval[:-1], interval[1:]):
        xy = np.where((dists <= thres_max) & (dists > thres_min))
        coord_lists.append(xy - c[:, np.newaxis])

    return coord_lists
