import json
import os
import shutil

import torch
from torch.nn import Parameter


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, epoch, is_best, save_dir="./outputs", file_name='checkpoint'):
    os.makedirs(save_dir, exist_ok=True)
    file = os.path.join(save_dir, f'{file_name}_{epoch}.pth.tar')
    torch.save(state, file)
    if is_best:
        shutil.copy(file, os.path.join(save_dir, 'model_best.pth.tar'))


def load_checkpoint(fpath):
    if os.path.isfile(fpath):
        checkpoint = torch.load(fpath)
        print(f"=> Loaded checkpoint '{fpath}'")
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def unfreeze_all_params(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)


def freeze_specific_params(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
