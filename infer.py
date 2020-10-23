import torch
from torch.utils.data import DataLoader
from torchvision.transforms import *
import cv2
import os

from datasets import DATASETS
from models import MODELS
from utils import load_checkpoint


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    if args.dataset not in DATASETS:
        raise Exception(f'`--dataset` is invalid. it should be one of {list(DATASETS.keys())}')
    infer_data = DATASETS[args.dataset](args.infer_root,
                                        transforms=Compose([ToTensor(),
                                                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                                        **args.__dict__)
    infer_loader = DataLoader(infer_data, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    net = MODELS[args.model_name](pretrained=False, **args.__dict__).to(device)
    checkpoint = load_checkpoint(args.model_file)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    for step, (file_path, img) in enumerate(infer_loader):
        img = img.to(device)
        pred, score_map = net(img)

        pred = pred.squeeze().type(torch.uint8).cpu().numpy()
        for i, file in enumerate(file_path):
            file_name = os.path.basename(file)
            file_name, _ = os.path.splitext(file_name)
            dst_file = os.path.join(args.save_dir, file_name + ".png")
            os.makedirs(args.save_dir, exist_ok=True)
            cv2.imwrite(dst_file, pred[i])

        print(f"process:{step + 1}/{len(infer_loader)}", end='\r', flush=True)
    print()


if __name__ == '__main__':
    from options.infer import parse_args

    main(parse_args())
