from campa.constants import EXPERIMENT_DIR
from campa.utils import init_logging
from campa.tl import Estimator, Experiment, Predictor, Cluster, ModelComparator
import argparse
import os

def prepare_exp_split(exp):
    """
    set up exp split data for non trainable model. Mimicks results folders created with predictor
    """
    from campa.data import MPPData
    import numpy as np
    # create results mpp_data for not trainable experiment to allow usage with Cluster
    for split in [exp.config['evaluation']['split'], exp.config['evaluation']['split']+'_imgs']:
        base_data_dir = os.path.join('datasets', exp.data_params['dataset_name'], split)
        mpp_params = {'base_data_dir':base_data_dir, 'subset': True}
        mpp_data = MPPData.from_data_dir(base_data_dir)
        if '_imgs' in split:
            # choose random img_ids from availalbe ones
            rng = np.random.default_rng(seed=42)
            img_ids = rng.choice(mpp_data.unique_obj_ids, exp.config['evaluation']['img_ids'], replace=False)
            # subset mpp_data to these img_ids
            mpp_data.subset(obj_ids=img_ids)
        mpp_data.write(save_dir=os.path.join(exp.full_path, 'results_epoch000', split), mpp_params=mpp_params, save_keys=[])


def run_experiments(mode, exps):
    exp_names = [exp.name for exp in exps]
    print(f'Running experiment for {exp_names} with mode {mode}')
    for exp_name, exp in zip(exp_names, exps):
        if mode in ('all', 'train', 'trainval'):
            if exp.is_trainable:
                print('Training model for {}'.format(exp_name))
                est = Estimator(exp)
                _ = est.train_model()
        if mode in ('all', 'evaluate', 'trainval'):
            if exp.is_trainable:
                # evaluate model
                print('Evaluating model for {}'.format(exp_name))
                pred = Predictor(exp)
                pred.evaluate_model()
            else:
                prepare_exp_split(exp)
            # cluster model
            print(f'Clustering results for {exp_name}')
            cl = Cluster.from_exp_split(exp)
            cl.create_clustering()
            # predict cluster for images
            if exp.config['evaluation']['predict_cluster_imgs']:
                cl.predict_cluster_imgs(exp)
    # compare models
    if mode in ('all', 'compare'):
        # assumes that all experiments have the same experiment_dir
        comp = ModelComparator(exps, save_dir=os.path.join(EXPERIMENT_DIR, exps[0].dir))
        comp.plot_history(values=['val_loss', 'val_decoder_loss'])
        comp.plot_final_score(score='val_decoder_loss', fallback_score='val_loss', save_prefix='decoder_loss_')
        comp.plot_per_channel_mse()
        comp.plot_predicted_images(img_ids=[0,1,2,3,4], img_size=exps[0].data_params['test_img_size'])
        comp.plot_cluster_images(img_ids=[0,1,2,3,4], img_size=exps[0].data_params['test_img_size'])
        comp.plot_umap()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=('Train and evaluate models defined by experiment config')
    )
    parser.add_argument('mode', default='all', choices=['all', 'train', 'evaluate', 'trainval', 'compare'])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='path_to_experiment_config.py')
    group.add_argument('--experiment-dir', help='experiment_dir containing experiment folders. Relative to EXPERIMENT_DIR')
    parser.add_argument('--exp-name', nargs='*', help='Select names of experiments to run. If not specified, all available experiments are run')
    return(parser.parse_args())


if __name__ == "__main__":
    args = parse_arguments()
    init_logging()
    if args.experiment_dir is not None:
        exps = Experiment.get_experiments_from_dir(args.experiment_dir, args.exp_name)
    else:
        exps = Experiment.get_experiments_from_config(config_fname=args.config)
    run_experiments(args.mode, exps)
