# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        data_folders = glob.glob(f'{data_path}/*/')
        self.data_list = [data for f in data_folders for data in glob.glob(f + '*.JPEG')]
        self.data_list += [data for f in data_folders for data in glob.glob(f + '*.jpg')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        img_size = image.size

        image = self.transform(image)
        image = np.array(image)[:, :, ::-1]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return {"image": image, "height": img_size[1], "width": img_size[0], 'image_path': image_path}


def collate_fn(batch):
    image_list = []
    for image in batch:
        image_list.append(image)
    return image_list
