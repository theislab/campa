from pathlib import Path
import os
import json
import random
import shutil
import string
import zipfile

import numpy as np
import pytest
import anndata as ad

from campa.constants import campa_config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def prepare_test_experiment(name, cluster_subset=False, full_data_prediction=False, full_data_clustering=False):
    """
    Copy reference_experiment with the specified components to name
    """
    from_dir = Path(campa_config.EXPERIMENT_DIR) / "reference_experiment/cVAE"
    to_dir = Path(campa_config.EXPERIMENT_DIR) / "test_experiment" / name
    # delete to_dir if it exists
    if to_dir.exists():
        shutil.rmtree(to_dir)
    to_dir.mkdir(parents=True, exist_ok=True)

    # copy model
    files_to_copy = []
    dirs_to_copy = []
    # copy model
    files_to_copy.extend(
        ["checkpoint", "config.json", "history.csv", "weights_epoch005.data-00000-of-00001", "weights_epoch005.index"]
    )
    dirs_to_copy.append("results_epoch005")
    if cluster_subset:
        dirs_to_copy.extend(["aggregated/sub-0.1"])
    if full_data_prediction:
        full_data_base = Path("aggregated/full_data")
        full_data_files = [
            "channels.csv",
            "latent.npy",
            "metadata.csv",
            "mpp_params.json",
            "obj_ids.npy",
            "x.npy",
            "y.npy",
        ]
        for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
            files_to_copy.extend([full_data_base / data_dir / f for f in full_data_files])
    if full_data_clustering:
        full_data_base = Path("aggregated/full_data")
        full_data_files = ["clustering.npy"]
        for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
            files_to_copy.extend([full_data_base / data_dir / f for f in full_data_files])

    # ensure dirs exist
    for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
        (to_dir / "aggregated/full_data" / data_dir).mkdir(parents=True, exist_ok=True)
    # copy files
    for f in files_to_copy:
        shutil.copy(from_dir / f, to_dir / f)
    # copy dirs
    for f in dirs_to_copy:
        shutil.copytree(from_dir / f, to_dir / f)

    # correct experiment dir + name in config.json
    config = json.load(open(to_dir / "config.json"))
    config["experiment"]["dir"] = "test_experiment"
    config["experiment"]["name"] = name
    json.dump(config, open(to_dir / "config.json", "w"), indent=4)


# -- FIXTURES ---
@pytest.fixture()
def _ensure_test_data():
    from campa.constants import SCRIPTS_DIR

    if os.path.isdir(os.path.join(SCRIPTS_DIR, "tests/_experiments")) and os.path.isdir(
        os.path.join(SCRIPTS_DIR, "tests/_data")
    ):
        return
    with zipfile.ZipFile(os.path.join(SCRIPTS_DIR, "tests/_test_data.zip"), "r") as zip_ref:
        zip_ref.extractall(os.path.join(SCRIPTS_DIR, "tests/."))


@pytest.fixture()
def test_experiment(_ensure_test_data):
    print(campa_config.EXPERIMENT_DIR)
    model_name = id_generator(size=6)
    prepare_test_experiment(model_name, cluster_subset=False, full_data_prediction=False, full_data_clustering=False)
    yield "test_experiment/" + model_name
    # remove experiment
    shutil.rmtree(os.path.join(campa_config.EXPERIMENT_DIR, "test_experiment", model_name))


@pytest.fixture()
def test_experiment_clustered(_ensure_test_data):
    model_name = id_generator(size=6)
    prepare_test_experiment(model_name, cluster_subset=True, full_data_prediction=False, full_data_clustering=False)
    yield "test_experiment/" + model_name
    # remove experiment
    shutil.rmtree(os.path.join(campa_config.EXPERIMENT_DIR, "test_experiment", model_name))


@pytest.fixture()
def test_experiment_full_data(_ensure_test_data):
    model_name = id_generator(size=6)
    prepare_test_experiment(model_name, cluster_subset=True, full_data_prediction=True, full_data_clustering=False)
    yield "test_experiment/" + model_name
    # remove experiment
    shutil.rmtree(os.path.join(campa_config.EXPERIMENT_DIR, "test_experiment", model_name))


@pytest.fixture()
def test_experiment_full_data_clustered(_ensure_test_data):
    model_name = id_generator(size=6)
    prepare_test_experiment(model_name, cluster_subset=True, full_data_prediction=True, full_data_clustering=True)
    yield "test_experiment/" + model_name
    # remove experiment
    shutil.rmtree(os.path.join(campa_config.EXPERIMENT_DIR, "test_experiment", model_name))


# --- HELPER FNS ---
def create_test_dataset():
    from campa.data import create_dataset

    # create test_dataset
    data_params = {
        "dataset_name": "test_dataset",
        "data_config": "TestData",
        "data_dirs": [
            os.path.join("184A1_unperturbed", well)
            for well in [
                "I09",
            ]
        ]
        + [
            os.path.join("184A1_meayamycin", well)
            for well in [
                "I12",
            ]
        ],
        "channels": [
            "01_PABPC1",
            "03_CDK9",
            "09_SRRM2",
            "10_POL2RA_pS2",
            "11_PML",
        ],
        "condition": ["perturbation_duration_one_hot", "cell_cycle_one_hot"],
        "condition_kwargs": {"cond_params": {}},
        "split_kwargs": {
            "train_frac": 0.35,
            "val_frac": 0.35,
        },
        "test_img_size": 225,
        "subset": True,
        "subset_kwargs": {"frac": None, "nona_condition": True, "cell_cycle": "NO_NAN"},
        "subsample": True,
        "subsample_kwargs": {
            "frac": None,
            "frac_per_obj": None,
            "num": None,
            "num_per_obj": 100,
        },
        "neighborhood": True,
        "neighborhood_size": 3,
        "normalise": True,
        "normalise_kwargs": {
            "background_value": "mean_background",
            "percentile": 98.0,
            "rescale_values": [],
        },
        "seed": 42,
    }
    create_dataset(data_params)


def train_test_model(model_name):
    from campa.tl import LossEnum, ModelEnum, Experiment, run_experiments

    experiment_config = {
        "experiment": {
            "dir": "test_experiment",
            "name": model_name,
            "save_config": True,
        },
        "data": {
            "data_config": "TestData",
            "dataset_name": "reference_dataset",
            "output_channels": None,
        },
        "model": {
            "model_cls": ModelEnum.VAEModel,
            "model_kwargs": {
                "num_neighbors": 3,
                "num_channels": 5,
                "num_output_channels": 5,
                "latent_dim": 4,
                # encoder definition
                "encoder_conv_layers": [16],
                "encoder_conv_kernel_size": [1],
                "encoder_fc_layers": [8],
                # decoder definition
                "decoder_fc_layers": [],
                "num_conditions": 6,
                "encode_condition": [6],
            },
            # if true, looks for saved weights in experiment_dir
            # if a path, loads these weights
            "init_with_weights": False,
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 128,
            "loss": {"decoder": LossEnum.SIGMA_MSE, "latent": LossEnum.KL},
            "metrics": {"decoder": LossEnum.MSE_metric, "latent": LossEnum.KL},
            # saving models
            "save_model_weights": True,
            "save_history": True,
            "overwrite_history": True,
        },
        "evaluation": {
            "split": "val",
            "predict_reps": ["latent", "decoder"],
            "img_ids": 1,
            "predict_imgs": True,
            "predict_cluster_imgs": True,
        },
        "cluster": {  # cluster config, also used in this format for whole data clustering
            "cluster_name": "clustering",
            "cluster_rep": "latent",
            "cluster_method": "leiden",  # leiden or kmeans
            "leiden_resolution": 0.2,
            "subsample": True,  # 'subsample' or 'som'
            "subsample_kwargs": {"frac": 0.1},
            "som_kwargs": {},
            "umap": True,
        },
    }

    exp = Experiment(experiment_config)
    run_experiments([exp], mode="trainval")


def extract_test_features(test_experiment):
    from campa.tl import extract_features

    feature_params = {
        "experiment_dir": test_experiment,
        "cluster_name": "clustering",
        "cluster_dir": "aggregated/sub-0.1",
        "cluster_col": None,
        "data_dirs": ["184A1_unperturbed/I09", "184A1_meayamycin/I12"],
        "save_name": "features.h5ad",
        "force": False,
        "features": ["intensity", "co-occurrence", "object-stats"],
        "co_occurrence_params": {
            "min": 2.0,
            "max": 10.0,
            "nsteps": 2,
            "logspace": False,
            "num_processes": 1,
        },
        "object_stats_params": {
            "features": ["area", "circularity", "elongation", "extent"],
            "channels": ["11_PML"],
        },
    }
    # 60s on MBP
    extract_features(feature_params)


# --- TESTS ---
def test_nn_dataset(_ensure_test_data):
    from campa.data import NNDataset

    create_test_dataset()
    test_ds = NNDataset("test_dataset", data_config="TestData")
    reference_ds = NNDataset("reference_dataset", data_config="TestData")

    # compare test and reference ds
    for split in ["train", "val", "test"]:
        print(split)
        print(test_ds.data[split]._compare(reference_ds.data[split]))
        assert test_ds.data[split]._compare(reference_ds.data[split])[0]
    for split in ["val", "test"]:
        assert test_ds.imgs[split]._compare(reference_ds.imgs[split])[0]


def test_model_training(_ensure_test_data):
    from campa.tl import Estimator, Experiment

    model_name = id_generator(size=6)
    train_test_model(model_name)

    # check if all expected files are created
    exp = Experiment.from_dir("test_experiment/" + model_name)
    exp.set_to_evaluate()
    _ = Estimator(exp)


def test_cluster_subset(test_experiment):
    from campa.tl import Cluster, create_cluster_data

    # cluster data
    create_cluster_data(test_experiment, subsample=True, frac=0.1, save_dir="aggregated/sub-0.1", cluster=True)

    # compare results
    test_cl = Cluster.from_cluster_data_dir(test_experiment + "/aggregated/sub-0.1")
    reference_cl = Cluster.from_cluster_data_dir("reference_experiment/cVAE/aggregated/sub-0.1")

    comp = test_cl.cluster_mpp._compare(reference_cl.cluster_mpp)[1]
    assert comp["x"]
    assert comp["y"]
    assert comp["obj_ids"]
    assert comp["mpp"]

    import numpy as np

    print(test_cl.cluster_mpp.data("latent"))
    print(reference_cl.cluster_mpp.data("latent"))
    # TODO why is latent not similar?
    print(np.isclose(test_cl.cluster_mpp.data("latent"), reference_cl.cluster_mpp.data("latent")))

    assert test_cl.cluster_annotation is not None
    test_cl.cluster_mpp.get_adata(X="mpp", obs=["clustering"], obsm={"X_latent": "latent", "X_umap": "umap"})


def test_predict_full_data(test_experiment_clustered):
    from campa.tl import prepare_full_dataset
    from campa.data import MPPData

    test_experiment = test_experiment_clustered
    # predict full data
    prepare_full_dataset(test_experiment, save_dir="aggregated/full_data")

    # check results
    for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
        # load mpp_data with cluster_rep
        test_mpp_data = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(campa_config.EXPERIMENT_DIR, test_experiment, "aggregated/full_data"),
            keys=["x", "y", "obj_ids", "latent"],
            data_config="TestData",
        )
        reference_mpp_data = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(campa_config.EXPERIMENT_DIR, test_experiment, "aggregated/full_data"),
            keys=["x", "y", "obj_ids", "latent"],
            data_config="TestData",
        )

        comp = test_mpp_data._compare(reference_mpp_data)[1]
        assert comp["x"]
        assert comp["y"]
        assert comp["obj_ids"]
        assert comp["mpp"]
        assert np.isclose(test_mpp_data.data("latent"), reference_mpp_data.data("latent")).all()


def test_cluster_full_data(test_experiment_full_data):
    from campa.tl import project_cluster_data
    from campa.data import MPPData

    test_experiment = test_experiment_full_data

    # predict full data
    project_cluster_data(
        test_experiment,
        cluster_data_dir="aggregated/sub-0.1",
        cluster_name="clustering",
        save_dir="aggregated/full_data",
    )

    # check results
    for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
        # load mpp_data with cluster_rep
        test_mpp_data = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(campa_config.EXPERIMENT_DIR, test_experiment, "aggregated/full_data"),
            keys=["x", "y", "obj_ids", "clustering"],
            data_config="TestData",
        )
        reference_mpp_data = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(campa_config.EXPERIMENT_DIR, test_experiment, "aggregated/full_data"),
            keys=["x", "y", "obj_ids", "clustering"],
            data_config="TestData",
        )

        assert test_mpp_data._compare(reference_mpp_data)[0]


def test_extract_features(test_experiment_full_data_clustered):
    from campa.tl import FeatureExtractor

    test_experiment = test_experiment_full_data_clustered
    extract_test_features(test_experiment)

    # check results
    for data_dir in ["184A1_unperturbed/I09", "184A1_meayamycin/I12"]:
        # load feature extractor from test data
        test_extr = FeatureExtractor.from_adata(
            os.path.join(
                campa_config.EXPERIMENT_DIR, test_experiment, "aggregated/full_data", data_dir, "features.h5ad"
            )
        )
        reference_extr = FeatureExtractor.from_adata(
            os.path.join(
                campa_config.EXPERIMENT_DIR, "reference_experiment/cVAE/aggregated/full_data", data_dir, "features.h5ad"
            )
        )
        assert test_extr._compare(reference_extr)[0]


def test_plot_intensity_features(_ensure_test_data):
    from campa.pl import (
        plot_mean_size,
        plot_mean_intensity,
        get_intensity_change,
        plot_intensity_change,
    )
    from campa.tl import Experiment, FeatureExtractor

    # use reference experiment for this. Just check that all plots are running without errors.
    # does not check content of plots.
    # load data
    exp = Experiment.from_dir("reference_experiment/cVAE")
    adatas = []
    for data_dir in exp.data_params["data_dirs"]:
        # get combined adata for dotplots
        extr = FeatureExtractor.from_adata(
            os.path.join(exp.full_path, "aggregated/full_data", data_dir, "features.h5ad")
        )
        adatas.append(extr.get_intensity_adata())
    adata_intensity = ad.concat(adatas, index_unique="-")

    plot_mean_intensity(
        adata_intensity,
        groupby="cluster",
        limit_to_groups={"perturbation": "normal"},
        dendrogram=False,
        layer=None,
        standard_scale="var",
        cmap="bwr",
        vmin=-4,
        vmax=4,
    )
    plot_mean_size(
        adata_intensity,
        groupby_row="cluster",
        groupby_col="perturbation_duration",
        normby_row="all",
        vmax=0.3,
    )

    res = get_intensity_change(
        adata_intensity,
        groupby="cluster",
        reference_group="perturbation_duration",
        reference=["normal"],
        limit_to_groups={"perturbation_duration": ["normal", "Meayamycin-720"]},
        color="logfoldchange",
        size="pval",
    )
    plot_intensity_change(**res, adjust_height=True, figsize=(15, 5), vmin=-2, vmax=2, dendrogram=True)

    res = get_intensity_change(
        adata_intensity,
        groupby="cluster",
        reference_group="perturbation_duration",
        reference=["normal"],
        limit_to_groups={"perturbation_duration": ["normal", "Meayamycin-720"]},
        color="logfoldchange",
        size="pval",
        norm_by_group="all",
    )
    plot_intensity_change(**res, adjust_height=True, figsize=(15, 5), vmin=-2, vmax=2)


def test_plot_co_occ_features(_ensure_test_data):
    from campa.pl import plot_co_occurrence, plot_co_occurrence_grid
    from campa.tl import Experiment, FeatureExtractor

    # use reference experiment for this. Just check that all plots are running without errors.
    # does not check content of plots.
    # load data
    exp = Experiment.from_dir("reference_experiment/cVAE")
    adatas = []
    for data_dir in exp.data_params["data_dirs"]:
        # get combined adata for dotplots
        extr = FeatureExtractor.from_adata(
            os.path.join(exp.full_path, "aggregated/full_data", data_dir, "features.h5ad")
        )
        adatas.append(extr.adata)
    adata_co_occ = ad.concat(adatas, index_unique="-", uns_merge="same")

    # plot meam co-occ scores
    condition = "perturbation_duration"
    condition_values = None

    # for one cluster-cluster pairing
    plot_co_occurrence(adata_co_occ, "0", "1", condition, condition_values)
    # for all cluster pairings
    fig, axes = plot_co_occurrence_grid(
        adata_co_occ, condition, condition_values, legend=False, ci=95, figsize=(20, 20)
    )


def test_plot_obj_stats(_ensure_test_data):
    from campa.pl import plot_object_stats
    from campa.tl import Experiment, FeatureExtractor

    # use reference experiment for this. Just check that all plots are running without errors.
    # does not check content of plots.
    # load data
    exp = Experiment.from_dir("reference_experiment/cVAE")
    adatas = []
    for data_dir in exp.data_params["data_dirs"]:
        # get combined adata for dotplots
        extr = FeatureExtractor.from_adata(
            os.path.join(exp.full_path, "aggregated/full_data", data_dir, "features.h5ad")
        )
        _ = extr.get_object_stats(area_threshold=10, agg=["median"])
        adatas.append(extr.adata)
    adata_object_stats = ad.concat(adatas, index_unique="-", uns_merge="same")

    # plot mean co-occ scores
    plot_object_stats(adata_object_stats, group_key="perturbation_duration", figsize_mult=(4, 4))
