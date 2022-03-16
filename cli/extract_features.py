import argparse
import logging
import os

import numpy as np

from campa.tl import Experiment, FeatureExtractor
from campa.utils import init_logging, load_config, merged_config


def extract_features(params):
    # set up FeatureExtractor
    log = logging.getLogger("extract_features")
    exp = Experiment.from_dir(params["experiment_dir"])
    data_dirs = params["data_dirs"]
    if data_dirs is None or len(data_dirs) == 0:
        data_dirs = exp.data_params["data_dirs"]

    for data_dir in data_dirs:
        log.info(f'extracting features {params["features"]} from {data_dir}')
        adata_fname = os.path.join(
            exp.full_path, "aggregated/full_data", data_dir, params["save_name"]
        )
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
            extr.extract_intensity_size(
                force=params["force"], fname=params["save_name"]
            )
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
                interval = np.linspace(
                    co_occ_params["min"], co_occ_params["max"], co_occ_params["nsteps"]
                ).astype(np.float32)
            extr.extract_co_occurrence(
                interval=interval, num_processes=co_occ_params["num_processes"]
            )
        if "object-stats" in params["features"]:
            obj_params = params["object_stats_params"]
            extr.extract_object_stats(
                features=obj_params["features"],
                intensity_channels=obj_params["channels"],
            )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("Extract features from clustered dataset. Created features adata.")
    )
    parser.add_argument("params", help="path to feature_params.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    init_logging()
    params = load_config(args.params)
    for variable_params in params.variable_feature_params:
        cur_params = merged_config(params.feature_params, variable_params)
        print(cur_params)
        extract_features(cur_params)
