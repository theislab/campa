import argparse
from miann.utils import init_logging
from miann.tl import Experiment, Cluster, Predictor
from miann.data import MPPData
import os
import logging

def prepare_full_dataset(args):
    log = logging.getLogger('prepare full dataset')
    exp = Experiment.from_dir(args.experiment_dir)
    # iterate over all data dirs
    for data_dir in exp.data_params['data_dirs']:
        log.info(f"Processing data_dir {data_dir}")
        mpp_data = MPPData.from_data_dir(data_dir)
        # params for partial saving of mpp_data
        mpp_params = {'base_data_dir':data_dir, 'subset': True}
        # prepare mpp_data
        log.info('Preparing data')
        mpp_data.prepare(exp.data_params)
        if exp.config['cluster']['cluster_rep'] == 'mpp':
            # just save mpp
            mpp_data.write(os.path.join(exp.full_path, args.save_dir, data_dir), mpp_params=mpp_params, save_keys=['mpp'])
        else:
            # need to predict rep - prepare neighborhood
            if exp.data_params['neighborhood']:
                mpp_data.add_neighborhood(exp.data_params['neighborhood_size'])
            # predict rep
            log.info('Predicting latent')
            pred = Predictor(exp)
            pred.predict(mpp_data, reps=[exp.config['cluster']['cluster_rep']], save_dir=os.path.join(exp.full_path, args.save_dir, data_dir), mpp_params=mpp_params)



def create_cluster_data(args):
    exp = Experiment.from_dir(args.experiment_dir)
    cluster_config = {'subsample': args.subsample, 'subsample_kwargs': {'frac': args.frac}}
    save_dir = args.save_dir if args.save_dir is not None else f'aggregated/sub-{args.frac}'
    cl = Cluster.from_exp(exp, cluster_config=cluster_config, data_dir=save_dir)
    # create cluster_mpp
    cl.create_cluster_mpp()
    # predict rep
    cl.predict_cluster_rep(exp)
    # get umap
    cl.add_umap()

def project_data(args):
    exp = Experiment.from_dir(args.experiment_dir)
    # set up cluster data
    cl = Cluster.from_cluster_data_dir(os.path.join(exp.dir, exp.name, args.cluster_data_dir))
    cl.set_cluster_name(args.cluster_name)
    assert cl.cluster_mpp.data(args.cluster_name) is not None, f"cluster data needs to contain clustering {args.cluster_name}"
    # iterate over all data dirs
    data_dirs = exp.data_params['data_dirs'] if args.data_dir is None else [args.data_dir]
    for data_dir in data_dirs:
        # load mpp_data with cluster_rep
        mpp_data = MPPData.from_data_dir(data_dir, base_dir=os.path.join(exp.full_path, args.save_dir), keys=['x', 'y', 'mpp', 'obj_ids', cl.config['cluster_rep']])
        cl.project_clustering(mpp_data, save_dir=os.path.join(exp.full_path, args.save_dir, data_dir))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cluster data and project clustering")
    parser.add_argument('experiment_dir', metavar='experiment-dir', help='relative to EXPERIMENT_PATH')
    subparsers = parser.add_subparsers(help='command to execute')

    # create
    create = subparsers.add_parser('create', help='Create (subsampled) data for clustering. Uses all data used to train exp')
    create.add_argument('--subsample', action='store_true', help='Subsample the data')
    create.add_argument('--frac', default=0.005, type=float, help='Fraction of pixels to use for clustering')
    create.add_argument('--save-dir', help='directory to save subsampled cluster data, relative to experiment dir. Default is aggregated/sub-FRAC')
    create.set_defaults(func=create_cluster_data)
    # prepare-full
    prepare = subparsers.add_parser('prepare-full', help='Prepare full data for clustering. Predicts cluster-rep.')
    prepare.add_argument('--save-dir', help='directory to save prepared full data to, relative to experiment dir.', default='aggregated/full_data')
    prepare.set_defaults(func=prepare_full_dataset)
    # project
    project = subparsers.add_parser('project', help='Project existing clustering')
    project.add_argument('cluster_data_dir', metavar='cluster-data-dir', help='directory in which clustering is stored relative to experiment dir. Usually in aggregated/sub-FRAC')
    project.add_argument('--save-dir', help='directory in which data to be projected is stored, relative to experiment dir.', default='aggregated/full_data')
    project.add_argument('--data-dir', help='data to project. If not specified, project all data_dirs in save_dir')
    project.add_argument('--cluster-name', default='clustering', help='name of clustering to project')
    project.set_defaults(func=project_data)

    return parser.parse_args(), parser

if __name__ == "__main__":
    args, parser = parse_arguments()
    init_logging()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)
