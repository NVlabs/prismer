# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob

from torch.utils.data import Dataset
from dataset.utils import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Caption(Dataset):
    def __init__(self, config, train=True):
        self.data_path = config['data_path']
        self.label_path = config['label_path']
        self.experts = config['experts']
        self.prefix = config['prefix']
        self.dataset = config['dataset']
        self.transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.0], train=train)
        self.train = train

        if train:
            self.data_list = []
            if self.dataset in ['coco', 'nocaps']:
                self.data_list += json.load(open(os.path.join(self.data_path, 'coco_karpathy_train.json'), 'r'))
        else:
            if self.dataset == 'coco':
                self.data_list = json.load(open(os.path.join(self.data_path, 'coco_karpathy_test.json'), 'r'))
            elif self.dataset == 'nocaps':
                self.data_list = json.load(open(os.path.join(self.data_path, 'nocaps_val.json'), 'r'))
            elif self.dataset == 'demo':
                data_folders = glob.glob(f'{self.data_path}/*/')
                self.data_list = [{'image': data} for f in data_folders for data in glob.glob(f + '*.jpg')]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        if self.dataset == 'coco':
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, data['image'], 'vqav2', self.experts)
        elif self.dataset == 'nocaps':
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, data['image'], 'nocaps', self.experts)
        elif self.dataset == 'demo':
            img_path_split = self.data_list[index]['image'].split('/')
            img_name = img_path_split[-2] + '/' + img_path_split[-1]
            image, labels, labels_info = get_expert_labels('', self.label_path, img_name, 'helpers', self.experts)

        experts = self.transform(image, labels)
        experts = post_label_process(experts, labels_info)

        if self.train:
            caption = pre_caption(self.prefix + ' ' + self.data_list[index]['caption'], max_words=30)
            return experts, caption
        else:
            return experts, index

