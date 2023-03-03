# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob

from torch.utils.data import Dataset
from dataset.utils import *


class Pretrain(Dataset):
    def __init__(self, config):
        self.cc12m_data_path = config['cc12m_data_path']
        self.cc3m_data_path = config['cc3m_data_path']
        self.coco_data_path = config['coco_data_path']
        self.vg_data_path = config['vg_data_path']
        self.label_path = config['label_path']
        self.experts = config['experts']

        self.data_list = []
        if 'cc12m' in config['datasets']:
            data_folders = glob.glob(f'{self.cc12m_data_path}/cc12m/*/')
            self.data_list += [{'image': data} for f in data_folders for data in glob.glob(f + '*.jpg')]
        if 'cc3m_sgu' in config['datasets']:
            data_folders = glob.glob(f'{self.cc3m_data_path}/cc3m_sgu/*/')
            self.data_list += [{'image': data} for f in data_folders for data in glob.glob(f + '*.jpg')]
        if 'coco' in config['datasets']:
            self.data_list += json.load(open(os.path.join(self.coco_data_path, 'coco_karpathy_train.json'), 'r'))
        if 'vg' in config['datasets']:
            self.data_list += json.load(open(os.path.join(self.vg_data_path, 'vg_caption.json'), 'r'))

        self.transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.5], train=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]['image']

        if 'cc12m' in img_path:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            image, labels, labels_info = get_expert_labels(self.cc12m_data_path, self.label_path, img_name, 'cc12m', self.experts)

            caption_path = img_path.replace('.jpg', '.txt')
            with open(caption_path) as f:
                caption = f.readlines()[0]
                
        elif 'cc3m_sgu' in img_path:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            image, labels, labels_info = get_expert_labels(self.cc3m_data_path, self.label_path, img_name, 'cc3m_sgu', self.experts)

            caption_path = img_path.replace('.jpg', '.txt')
            with open(caption_path) as f:
                caption = f.readlines()[0]

        elif 'train2014' in img_path or 'val2014' in img_path:
            image, labels, labels_info = get_expert_labels(self.coco_data_path, self.label_path, img_path, 'vqav2', self.experts)
            caption = self.data_list[index]['caption']

        elif 'visual-genome' in img_path:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            image, labels, labels_info = get_expert_labels(self.vg_data_path, self.label_path, img_name, 'vg', self.experts)
            caption = self.data_list[index]['caption']

        experts = self.transform(image, labels)
        experts = post_label_process(experts, labels_info)
        caption = pre_caption(caption, max_words=30)
        return experts, caption
