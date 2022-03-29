import sys
import argparse

from campa.utils import load_config, init_logging, merged_config
from campa.cli.prepare_config import prepare_config
import campa
import campa.tl


class CAMPA:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="CAMPA - conditional autoencoder for multiplexed image analysis",
            usage="""campa <command> [<args>]

Available subcommands are:
    setup               Create configuration file campa.ini
    create_dataset      Create NNDataset using parameter file
    train               Train and evaluate models defined by experiment config
    cluster             Cluster data and project clustering
    extract_features    Extract features from clustered dataset
""",
        )
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
        # in subcommands, irgnore first TWO argvs (campa) and the subcommand

    def setup(self):
        parser = argparse.ArgumentParser(description="Create configuration file campa.ini")
        parser.add_argument(
            "--force",
            action="store_true",
            help="force recreation of campa.ini even if exists",
        )
        args = parser.parse_args(sys.argv[2:])
        # create and populate campa.ini
        prepare_config(args)

    def create_dataset(self):
        parser = argparse.ArgumentParser(description=("Create NNDataset using parameter file"))
        parser.add_argument("params", help="path to data_params.py")
        args = parser.parse_args(sys.argv[2:])
        init_logging()
        params = load_config(args.params)
        campa.data.create_dataset(params.data_params)

    def train(self):
        parser = argparse.ArgumentParser(description=("Train and evaluate models defined by experiment config"))
        parser.add_argument(
            "mode",
            default="all",
            choices=["all", "train", "evaluate", "trainval", "compare"],
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--config", help="path_to_experiment_config.py")
        group.add_argument(
            "--experiment-dir",
            help="experiment_dir containing experiment folders. Relative to EXPERIMENT_DIR",
        )
        parser.add_argument(
            "--exp-name",
            nargs="*",
            help="Select names of experiments to run. If not specified, all available experiments are run",
        )
        args = parser.parse_args(sys.argv[2:])
        init_logging()
        if args.experiment_dir is not None:
            exps = campa.tl.Experiment.get_experiments_from_dir(args.experiment_dir, args.exp_name)
        else:
            exps = campa.tl.Experiment.get_experiments_from_config(config_fname=args.config)
            # NOTE exp_name is not respected here!
        campa.tl.run_experiments(exps=exps, mode=args.mode)

    def cluster(self):
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
            help="directory to save subsampled cluster data, relative to experiment dir. Default is aggregated/sub-FRAC",  # noqa: E501
        )
        create.add_argument(
            "--cluster",
            help="use cluster params in Experiment config to cluster the subsetted data.",
            action="store_true",
        )
        create.set_defaults(func=campa.tl.create_cluster_data)
        # prepare-full
        prepare = subparsers.add_parser("prepare-full", help="Prepare full data for clustering. Predicts cluster-rep.")
        prepare.add_argument(
            "--save-dir",
            help="directory to save prepared full data to, relative to experiment dir.",
            default="aggregated/full_data",
        )
        prepare.set_defaults(func=campa.tl.prepare_full_dataset)
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
        project.set_defaults(func=campa.tl.project_cluster_data)

        args = parser.parse_args(sys.argv[2:])
        init_logging()
        try:
            func = args.func
        except AttributeError:
            parser.error("too few arguments")
        func(**vars(args))

    def extract_features(self):
        parser = argparse.ArgumentParser(description=("Extract features from clustered dataset"))
        parser.add_argument("params", help="path to feature_params.py")
        args = parser.parse_args(sys.argv[2:])
        init_logging()
        params = load_config(args.params)
        for variable_params in params.variable_feature_params:
            cur_params = merged_config(params.feature_params, variable_params)
            print(cur_params)
            campa.tl.extract_features(cur_params)


def main():
    CAMPA()


if __name__ == "__main__":
    CAMPA()
