from datasets import DATASETS
import argparse

from models import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for testing, which is one of {}'.format(str(list(MODELS.keys()))),
        type=str,
        default='FarNet')

    # params of infer
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help="The dataset you want to test, which is one of {}".format(str(list(DATASETS.keys()))),
        type=str,
        default='Map')
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help="dataset root directory",
        type=str,
        default=None)

    # params of prediction
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[256, 256],
        type=int)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=32)

    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The path of model for evaluation',
        type=str,
        default=None)

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output/result')

    return parser.parse_args()
