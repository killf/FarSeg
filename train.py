import torch
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import MODELS
from transforms import *


def collate_fn(x):
    return x


def main(args):
    device = torch.device(args.device)

    if args.dataset not in DATASETS:
        raise Exception(f'`--dataset` is invalid. it should be one of {list(DATASETS.keys())}')
    train_data = DATASETS[args.dataset](args.train_root,
                                        transforms=Compose([RandomCrop(224),
                                                            RandomHorizontalFlip(),
                                                            RandomVerticalFlip(),
                                                            ToTensor(),
                                                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                                        **args.__dict__)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    do_val = False
    if args.val_root:
        val_data = DATASETS[args.dataset](args.val_root, transforms=ToTensor(), **args.__dict__)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
        do_val = True

    net = MODELS[args.model_name](pretrained=True, **args.__dict__).to(device)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    for epoch in range(100):
        net.train()
        for step, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            loss = net(img, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch:{epoch} Step:{step} Loss:{float(loss):04f}", end="\r", flush=True)
        print()
        pass

    pass


if __name__ == '__main__':
    from options.train import parse_args

    main(parse_args())
