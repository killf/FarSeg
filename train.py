import torch
from torch.utils.data import DataLoader

from datasets import DATASETS
from models import MODELS
from transforms import *
from utils import Timer, Counter, calculate_eta


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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    do_val = False
    if args.val_root:
        val_data = DATASETS[args.dataset](args.val_root, transforms=ToTensor(), **args.__dict__)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)
        do_val = True

    net = MODELS[args.model_name](pretrained=True, **args.__dict__).to(device)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    start_epoch, total_epoch = 0, args.epochs

    for epoch in range(start_epoch, total_epoch):
        net.train()

        timer, counter = Timer(), Counter()
        timer.start()
        for step, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            reader_time = timer.elapsed_time()

            loss, miou = net(img, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            batch_time = timer.elapsed_time()
            counter.append(loss=loss, miou=miou, reader_time=reader_time, batch_time=batch_time)
            eta = calculate_eta(len(train_loader) - step, counter.batch_time)
            print(f"[epoch={epoch + 1}/{total_epoch}] "
                  f"[step={step + 1}/{len(train_loader)}] "
                  f"loss={loss:.4f}/{counter.loss:.4f} "
                  f"miou={miou:.4f}/{counter.miou:.4f} "
                  f"batch_time={counter.batch_time:.4f} "
                  f"reader_time={counter.reader_time:.4f} "
                  f"| ETA {eta}",
                  end="\r",
                  flush=True)
            timer.restart()
        print()
        pass

    pass


if __name__ == '__main__':
    from options.train import parse_args

    main(parse_args())
