# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import os
import PIL.Image as Image
import numpy as np
import cv2
import clip
import pickle as pk
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

from experts.model_bank import load_expert_model
from experts.ocr_detection.generate_dataset import Dataset
from accelerate import Accelerator
from tqdm import tqdm


model, transform = load_expert_model(task='ocr_detection')
accelerator = Accelerator(mixed_precision='fp16')
pca_clip = pk.load(open('dataset/clip_pca.pkl', 'rb'))

config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
data_path = config['data_path']
save_path = os.path.join(config['save_path'], 'ocr_detection')

batch_size = 32
dataset = Dataset(data_path, transform)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

clip_model, _ = clip.load("ViT-L/14", device=accelerator.device)
model, data_loader = accelerator.prepare(model, data_loader)


def get_label(w, h, word_instances):
    word_lists = []
    final_mask = np.ones([h, w],  dtype=np.uint8) * 255
    counter = 0
    for word_instance in word_instances[::-1]:
        mask = np.zeros([h ,w])
        mask = cv2.fillPoly(mask, [np.int32(word_instance.word_bbox.reshape(-1, 2))], 1)
        text = word_instance.text.lower()
        final_mask[mask > 0] = counter
        word_lists.append(text)
        counter += 1
    return final_mask, word_lists


with torch.no_grad():
    for i, (test_data, image_path, scale_w, scale_h, original_w, original_h) in enumerate(tqdm(data_loader)):
        word_instance_lists = model(test_data, scale_w, scale_h, original_w, original_h)
        for k in range(len(word_instance_lists)):
            word_instance = word_instance_lists[k]
            if len(word_instance) == 0:
                continue
            else:
                final_mask, word_lists = get_label(original_w[k], original_h[k], word_instance)

            final_mask = Image.fromarray(final_mask)
            img_path_split = image_path[k].split('/')
            ps = image_path[k].split('.')[-1]
            im_save_path = os.path.join(save_path, img_path_split[-3], img_path_split[-2])
            os.makedirs(im_save_path, exist_ok=True)

            final_mask.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))

            if len(word_lists) > 0:
                word_embed = clip.tokenize(word_lists).to(accelerator.device)
                word_features = pca_clip.transform(clip_model.encode_text(word_embed).float().cpu())
                word_lists = {j: {'features': torch.from_numpy(word_features[j]).float(),
                                  'text': word_lists[j]} for j in range(len(word_lists))}
                torch.save(word_lists, os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.pt')))


