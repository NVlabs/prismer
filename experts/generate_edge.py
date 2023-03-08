# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

from experts.model_bank import load_expert_model
from experts.edge.generate_dataset import Dataset
from experts.edge.images import fuse_edge
import PIL.Image as Image
from accelerate import Accelerator
from tqdm import tqdm


model, transform = load_expert_model(task='edge')
accelerator = Accelerator(mixed_precision='fp16')

config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
data_path = config['data_path']
save_path = os.path.join(config['save_path'], 'edge')

batch_size = 64
dataset = Dataset(data_path, transform)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model, data_loader = accelerator.prepare(model, data_loader)

with torch.no_grad():
    for i, (test_data, img_path, img_size) in enumerate(tqdm(data_loader)):
        test_pred = model(test_data)
        fuses = fuse_edge(test_pred)
        for k in range(len(fuses)):
            edge = fuses[k]
            img_path_split = img_path[k].split('/')
            ps = img_path[k].split('.')[-1]
            im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
            os.makedirs(im_save_path, exist_ok=True)

            im_size = img_size[0][k].item(), img_size[1][k].item()
            edge = Image.fromarray(edge).convert('L')
            edge = edge.resize((im_size[0], im_size[1]), resample=Image.Resampling.BILINEAR)
            edge.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))


