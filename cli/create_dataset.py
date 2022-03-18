import argparse

from campa.data import create_dataset
from campa.utils import load_config, init_logging


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Create NNDataset using either params file"))
    parser.add_argument("params", help="path to data_params.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    init_logging()
    params = load_config(args.params)
    create_dataset(params.data_params)
