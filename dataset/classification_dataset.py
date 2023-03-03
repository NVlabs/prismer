# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob
from torch.utils.data import Dataset
from dataset.utils import *


class Classification(Dataset):
    def __init__(self, config, train):
        self.data_path = config['data_path']
        self.label_path = config['label_path']
        self.experts = config['experts']
        self.dataset = config['dataset']
        self.shots = config['shots']
        self.prefix = config['prefix']

        self.train = train
        self.transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.0], train=True)

        if train:
            data_folders = glob.glob(f'{self.data_path}/imagenet_train/*/')
            self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.JPEG')[:self.shots]]
            self.answer_list = json.load(open(f'{self.data_path}/imagenet/' + 'imagenet_answer.json'))
            self.class_list = json.load(open(f'{self.data_path}/imagenet/' + 'imagenet_class.json'))
        else:
            data_folders = glob.glob(f'{self.data_path}/imagenet/*/')
            self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.JPEG')]
            self.answer_list = json.load(open(f'{self.data_path}/imagenet/' + 'imagenet_answer.json'))
            self.class_list = json.load(open(f'{self.data_path}/imagenet/' + 'imagenet_class.json'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]['image']
        if self.train:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            class_name = img_path_split[-2]
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, img_name, 'imagenet_train', self.experts)
        else:
            img_path_split = img_path.split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            class_name = img_path_split[-2]
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, img_name, 'imagenet', self.experts)

        experts = self.transform(image, labels)
        experts = post_label_process(experts, labels_info)

        if self.train:
            caption = self.prefix + ' ' + self.answer_list[int(self.class_list[class_name])].lower()
            return experts, caption
        else:
            return experts, self.class_list[class_name]





# import os
# import glob
#
# data_path = '/Users/shikunliu/Documents/dataset/mscoco/mscoco'
#
# data_folders = glob.glob(f'{data_path}/*/')
# data_list = [data for f in data_folders for data in glob.glob(f + '*.jpg')]


