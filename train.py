import torch
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import MODELS
from transforms import *


def main(args):
    device = torch.device(args.device)

    if args.dataset not in DATASETS:
        raise Exception(f'`--dataset` is invalid. it should be one of {list(DATASETS.keys())}')
    train_data = DATASETS[args.dataset](args.train_root, transforms=Compose([RandomCrop(224),
                                                                             RandomHorizontalFlip(),
                                                                             RandomVerticalFlip(),
                                                                             RandomErasing(),
                                                                             ToTensor()]), **args.__dict__)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    do_val = False
    if args.val_root:
        val_data = DATASETS[args.dataset](args.val_root, transforms=ToTensor(), **args.__dict__)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
        do_val = True

    pass


if __name__ == '__main__':
    from options.train import parse_args

    main(parse_args())
