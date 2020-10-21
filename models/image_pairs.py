from torch.utils.data import Dataset
from PIL import Image

import os


class ImagePairs(Dataset):
    def __init__(self, root, image_folder="image", label_folder="label", transforms=None, **kwargs):
        self.root = root
        self.image_folder = image_folder if os.path.isabs(image_folder) else os.path.join(root, image_folder)
        self.label_folder = label_folder if os.path.isabs(label_folder) else os.path.join(root, label_folder)
        self.transforms = transforms

        img_files = dict(os.path.splitext(file_name) for file_name in os.listdir(self.image_folder))
        grt_files = dict(os.path.splitext(file_name) for file_name in os.listdir(self.label_folder))

        self.file_list = []
        for img_filename, img_ext in img_files.items():
            if img_filename not in grt_files:
                continue

            img_path = os.path.join(self.image_folder, img_filename + img_ext)
            grt_path = os.path.join(self.label_folder, img_filename + grt_files[img_filename])
            self.file_list.append((img_path, grt_path))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path, grt_path = self.file_list[index]
        img = Image.open(img_path)
        grt = Image.open(grt_path)
        if self.transforms:
            img, grt = self.transforms(img, grt)
        return img, grt
