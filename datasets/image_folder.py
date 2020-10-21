from torch.utils.data import Dataset
from PIL import Image

import os


class ImageFolder(Dataset):
    def __init__(self, root, transforms=None, **kwargs):
        self.root = root
        self.transforms = transforms

        self.file_list = [os.path.join(root, file) for file in os.listdir(root)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        img = Image.open(file_path)
        if self.transforms:
            img = self.transforms(img)
        return img
