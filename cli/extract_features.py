import argparse
from miann.utils import init_logging
from miann.tl import Experiment, Cluster, Predictor
from miann.data import MPPData
import os
import logging
from miann.tl import FeatureExtractor, Experiment

def extract_features(args):
    # set up FeatureExtractor
    log = logging.getLogger('extract_features')
    exp = Experiment.from_dir(args.experiment_dir)
    for data_dir in exp.data_params['data_dirs']:
        log.info(f'extracting features {args.mode} from {data_dir}')
        adata_fname = os.path.join(exp.full_path, 'aggregated/full_data', data_dir, args.save_name)
        if os.path.exists(adata_fname):
            log.info(f'initialising from existing adata {adata_fname}')
            extr = FeatureExtractor.from_adata(adata_fname)
        else:
            extr = FeatureExtractor(exp, data_dir=data_dir, cluster_name=args.cluster_name, cluster_dir=args.cluster_dir, cluster_col=args.cluster_col)
        # extract features
        if 'intensity' in args.mode:
            extr.extract_intensity_size(force=args.force, fname=args.save_name)
        # TODO add more features here (spatial co-occurence + blob counts)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=('Extract features from clustered dataset. Created features adata.')
    )

    parser.add_argument('experiment_dir', metavar='experiment-dir', help='relative to EXPERIMENT_DIR')
    parser.add_argument('cluster_name', metavar='cluster-name', help='name of clustering to use')
    parser.add_argument('--cluster-dir', help='dir of subsampled clustering to load annotation. Relative to experiment_dir. Default is taking first of experiment_dir/aggregated/sub-*')
    parser.add_argument('--cluster-col', help='cluster annotation to use. Defaults to cluster_name')
    parser.add_argument('--save-name', default='features.h5ad', help='filename to use for saving extracted features. Default is features.h5ad')
    parser.add_argument('--force', action='store_true', help='force calculation even when adata exists')
    parser.add_argument('mode', nargs="+", choices=['intensity'], help='type of features to extract. Intensity: per-cluster mean and size features. Use this first to set up the adata.')
    
    return(parser.parse_args())


if __name__ == "__main__":
    args = parse_arguments()
    init_logging()
    extract_features(args)
