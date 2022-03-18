from typing import Optional
import os
import logging
import argparse

from campa.tl import Cluster, Predictor, Experiment
from campa.data import MPPData
from campa.utils import init_logging


def prepare_full_dataset(experiment_dir: str, save_dir: str = "aggregated/full_data"):
    """
    Prepare all data for clustering by predicting cluster-rep.

    Parameters
    ----------
    experiment_dir
        experiment directory releative to EXPERIMENT_PATH
    save_dir
        directory to save prepared full data to, relative to experiment dir
    """
    log = logging.getLogger("prepare full dataset")
    exp = Experiment.from_dir(experiment_dir)
    # iterate over all data dirs
    for data_dir in exp.data_params["data_dirs"]:
        log.info(f"Processing data_dir {data_dir}")
        mpp_data = MPPData.from_data_dir(data_dir)
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


def create_cluster_data(experiment_dir, subsample: bool = False, frac: float = 0.005, save_dir=None):
    """
    Create (subsampled) data for clustering.

    Uses dataset used to train experiment.

    Parameters
    ----------
    experiment_dir
        experiment directory releative to EXPERIMENT_PATH
    subsample
        subsample the data
    frac
        Fraction of pixels to use for clustering if subsample is True
    save_dir
        directory to save subsampled cluster data, relative to experiment dir.
        default is aggregated/sub-FRAC
    """
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
    # get umap
    cl.add_umap()


def project_data(
    experiment_dir: str,
    cluster_data_dir: str,
    cluster_name: str = "clustering",
    save_dir: str = "aggregated/full_data",
    data_dir: Optional[str] = None,
):
    """
    Project existing clustering to new data

    Parameters
    ----------
    experiment_dir
        experiment directory releative to EXPERIMENT_PATH
    cluster_data_dir
        directory in which clustering is stored relative to experiment dir. Usually in aggregated/sub-FRAC
    cluster_name
        name of clustering to project
    save_dir
        directory in which the data to be projected is stored, relative to experiment dir.
    data_dir
        data_dir to project. If not specified, project all data_dirs in save_dir.
        Relative to save_dir
    """
    exp = Experiment.from_dir(experiment_dir)
    # set up cluster data
    cl = Cluster.from_cluster_data_dir(os.path.join(exp.dir, exp.name, cluster_data_dir))
    cl.set_cluster_name(cluster_name)
    assert cl.cluster_mpp.data(cluster_name) is not None, f"cluster data needs to contain clustering {cluster_name}"
    # iterate over all data dirs
    data_dirs = exp.data_params["data_dirs"] if data_dir is None else [data_dir]
    for data_dir in data_dirs:
        # load mpp_data with cluster_rep
        mpp_data = MPPData.from_data_dir(
            data_dir,
            base_dir=os.path.join(exp.full_path, save_dir),
            keys=["x", "y", "obj_ids", cl.config["cluster_rep"]],
        )
        cl.project_clustering(mpp_data, save_dir=os.path.join(exp.full_path, save_dir, data_dir))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cluster data and project clustering")
    parser.add_argument("experiment_dir", metavar="experiment-dir", help="relative to EXPERIMENT_PATH")
    subparsers = parser.add_subparsers(help="command to execute")

    # create
    create = subparsers.add_parser(
        "create",
        help="Create (subsampled) data for clustering. Uses all data used to train exp",
    )
    create.add_argument("--subsample", action="store_true", help="Subsample the data")
    create.add_argument(
        "--frac",
        default=0.005,
        type=float,
        help="Fraction of pixels to use for clustering",
    )
    create.add_argument(
        "--save-dir",
        help="directory to save subsampled cluster data, relative to experiment dir. Default is aggregated/sub-FRAC",
    )
    create.set_defaults(func=create_cluster_data)
    # prepare-full
    prepare = subparsers.add_parser("prepare-full", help="Prepare full data for clustering. Predicts cluster-rep.")
    prepare.add_argument(
        "--save-dir",
        help="directory to save prepared full data to, relative to experiment dir.",
        default="aggregated/full_data",
    )
    prepare.set_defaults(func=prepare_full_dataset)
    # project
    project = subparsers.add_parser("project", help="Project existing clustering")
    project.add_argument(
        "cluster_data_dir",
        metavar="cluster-data-dir",
        help="directory in which clustering is stored relative to experiment dir. Usually in aggregated/sub-FRAC",
    )
    project.add_argument(
        "--save-dir",
        help="directory in which data to be projected is stored, relative to experiment dir.",
        default="aggregated/full_data",
    )
    project.add_argument(
        "--data-dir",
        help="data to project. If not specified, project all data_dirs in save_dir",
    )
    project.add_argument("--cluster-name", default="clustering", help="name of clustering to project")
    project.set_defaults(func=project_data)

    return parser.parse_args(), parser


if __name__ == "__main__":
    args, parser = parse_arguments()
    init_logging()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(**vars(args))
