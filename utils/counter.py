import torch
import numpy as np
import time


class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()

            self.data[key].append(value)

    def __getattr__(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])
