from miann.constants import EXPERIMENT_DIR
from miann.utils import init_logging, load_config
from miann.tl import Estimator, Experiment, Predictor, Cluster
from miann.utils import merged_config
import argparse
import os
import json

def get_exps(config, exp_names=None):
    """
    init and return experiments from configs in config.py file
    """
    config = load_config(config)
    exps = []
    for exp_config in config.variable_config:
        cur_config = merged_config(config.base_config, exp_config)
        if exp_names is None or cur_config['experiment']['name'] in exp_names:
            exps.append(Experiment(cur_config))
    return exps

def get_exps_from_dir(exp_dir, exp_names=None):
    exps = []
    for exp_name in next(os.walk(os.path.join(EXPERIMENT_DIR, exp_dir)))[1]:
        config_fname = os.path.join(EXPERIMENT_DIR, exp_dir, exp_name, 'config.json')
        if os.path.exists(config_fname) and exp_name in exp_names:
            exps.append(Experiment.from_dir(os.path.join(exp_dir, exp_name)))
    return exps

def run_experiments(mode, exps):
    exp_names = [exp.name for exp in exps]
    print(f'Running experiment for {exp_names} with mode {mode}')
    for exp_name, exp in zip(exp_names, exps):
        if mode in ('all', 'train', 'trainval'):
            print('Training model for {}'.format(exp_name))
            est = Estimator(exp)
            _ = est.train_model()
        if mode in ('all', 'evaluate', 'trainval'):
            # evaluate model
            print('Evaluating model for {}'.format(exp_name))
            pred = Predictor(exp)
            pred.evaluate_model()
            # cluster model
            print(f'Clustering results for {exp_name}')
            cl = Cluster.from_exp_split(exp)
            cl.create_clustering()
            # predict cluster for images
            if exp.config['evaluate']['predict_cluster_imgs']:
                cl.predict_cluster_imgs(exp)
    # compare models
    #if mode in ('all', 'compare'):
        # assumes that all experiments have the same experiment_dir
        #compare_models(exp_names, configs[0]['experiment']['experiment_dir'])

        
#def compare_models(exp_names, experiment_dir):
#    print('Comparing models {}'.format(exp_names))
#    evas = {exp_name:Evaluator.from_evaluated(exp_name, experiment_dir, load_images=True) for exp_name in exp_names}
#    comp = ModelComparator(evas, save_dir=os.path.join(EXPERIMENT_DIR, experiment_dir))
#    comp.plot_history(values=['val_loss', 'val_decoder_loss'])
#    comp.plot_final_score(score='val_decoder_loss', fallback_score='val_loss', save_prefix='decoder_loss_')
#    comp.plot_per_channel_mse()
#    comp.plot_predicted_images(img_ids=[0,1,2,3,4])
#    comp.plot_cluster_images(img_ids=[0,1,2,3,4])
#    comp.plot_cluster_heatmap()
    #comp.plot_cluster_overlap(compare_with=['well_name', 'perturbation'])


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
        exps = get_exps_from_dir(args.experiment_dir, args.exp_name)
    else:
        exps = get_exps(config=args.config)
    run_experiments(args.mode, exps)
