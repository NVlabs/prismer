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
from experts.normal.generate_dataset import CustomDataset
import PIL.Image as Image
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np


model, transform = load_expert_model(task='normal')
accelerator = Accelerator(mixed_precision='fp16')

config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
data_path = config['data_path']
save_path = os.path.join(config['save_path'], 'normal')

batch_size = 64
dataset = CustomDataset(data_path, transform)
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
        pred_norm = test_pred[0][-1][:, :3]
        for k in range(len(pred_norm)):
            img_path_split = img_path[k].split('/')
            ps = img_path[k].split('.')[-1]
            im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
            os.makedirs(im_save_path, exist_ok=True)

            im_size = img_size[0][k].item(), img_size[1][k].item()
            norm = pred_norm[k]
            norm = ((norm + 1) * 0.5).clip(0, 1)
            norm = torch.nn.functional.interpolate(norm.unsqueeze(0), size=(im_size[1], im_size[0]), mode='bilinear', align_corners=True)
            norm_im = Image.fromarray((norm[0] * 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)).convert('RGB')
            norm_im.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))


