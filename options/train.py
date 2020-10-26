from datasets import DATASETS
import argparse

from models import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for training, which is one of {}'.format(str(list(MODELS.keys()))),
        type=str,
        default='FarNet')

    # params of dataset
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help="The dataset you want to train, which is one of {}".format(str(list(DATASETS.keys()))),
        type=str,
        default='ImagePairs')
    parser.add_argument(
        '--train_root',
        dest='train_root',
        help="train dataset root directory",
        type=str,
        required=True)
    parser.add_argument(
        '--val_root',
        dest='val_root',
        help="val dataset root directory",
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help="number works of data loader",
        type=int,
        default=0)
    parser.add_argument(
        '--device',
        dest='device',
        help='device for training',
        type=str,
        default="cuda")

    # params of training
    parser.add_argument(
        '--epochs',
        dest='epochs',
        help='epochs for training',
        type=int,
        default=20)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=32)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='Learning rate',
        type=float,
        default=0.0005)
    parser.add_argument(
        '--resume',
        dest='resume',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./outputs')
    parser.add_argument(
        '--logs_dir',
        dest='logs_dir',
        help='The directory for saving the log message',
        type=str,
        default='./logs')

    return parser.parse_args()
