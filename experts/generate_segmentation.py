# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import os
import PIL.Image as Image
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
    
from experts.model_bank import load_expert_model
from experts.segmentation.generate_dataset import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm

model, transform = load_expert_model(task='seg_coco')
accelerator = Accelerator(mixed_precision='fp16')

config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
data_path = config['data_path']
save_path = os.path.join(config['save_path'], 'seg_coco')

batch_size = 4
dataset = Dataset(data_path, transform)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)


model, data_loader = accelerator.prepare(model, data_loader)

with torch.no_grad():
    for i, test_data in enumerate(tqdm(data_loader)):
        test_pred = model(test_data)

        for k in range(len(test_pred)):
            pred = test_pred[k]['sem_seg']
            labels = torch.argmax(pred, dim=0)

            img_path_split = test_data[k]['image_path'].split('/')
            ps = test_data[k]['image_path'].split('.')[-1]
            im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
            os.makedirs(im_save_path, exist_ok=True)

            seg = Image.fromarray(labels.float().detach().cpu().numpy()).convert('L')
            seg.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))

