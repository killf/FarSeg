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
        default='ImageFolder')
    parser.add_argument(
        '--infer_root',
        dest='infer_root',
        help="dataset root directory",
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help="number works of data loader",
        type=int,
        default=0)

    # params of prediction
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=32)
    parser.add_argument(
        '--model_file',
        dest='model_file',
        help='The path of model for evaluation',
        type=str,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./outputs/result')
    parser.add_argument(
        '--device',
        dest='device',
        help='device for training',
        type=str,
        default="cuda")

    return parser.parse_args()
