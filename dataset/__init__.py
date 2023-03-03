# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

from torch.utils.data import DataLoader

from dataset.pretrain_dataset import Pretrain
from dataset.vqa_dataset import VQA
from dataset.caption_dataset import Caption
from dataset.classification_dataset import Classification


def create_dataset(dataset, config):
    if dataset == 'pretrain':
        dataset = Pretrain(config)
        return dataset

    elif dataset == 'vqa':
        train_dataset = VQA(config, train=True)
        test_dataset = VQA(config, train=False)
        return train_dataset, test_dataset

    elif dataset == 'caption':
        train_dataset = Caption(config, train=True)
        test_dataset = Caption(config, train=False)
        return train_dataset, test_dataset
    
    elif dataset == 'classification':
        train_dataset = Classification(config, train=True)
        test_dataset = Classification(config, train=False)
        return train_dataset, test_dataset


def create_loader(dataset, batch_size, num_workers, train, collate_fn=None):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             shuffle=True if train else False,
                             drop_last=True if train else False)
    return data_loader
