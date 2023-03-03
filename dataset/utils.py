# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import os
import re
import json
import torch
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from dataset.randaugment import RandAugment

COCO_FEATURES = torch.load('dataset/coco_features.pt')['features']
ADE_FEATURES = torch.load('dataset/ade_features.pt')['features']
DETECTION_FEATURES = torch.load('dataset/detection_features.pt')['features']
BACKGROUND_FEATURES = torch.load('dataset/background_features.pt')


class Transform:
    def __init__(self, resize_resolution=384, scale_size=[0.5, 1.0], train=False):
        self.resize_size = [resize_resolution, resize_resolution]
        self.scale_size = scale_size
        self.train = train
        self.randaugment = RandAugment(2, 5)

    def __call__(self, image, labels):
        if self.train:
            # random resize crop
            i, j, h, w = transforms.RandomResizedCrop.get_params(img=image, scale=self.scale_size, ratio=[3. / 4, 4. / 3])
            image = transforms_f.crop(image, i, j, h, w)
            if labels is not None:
                for exp in labels:
                    labels[exp] = transforms_f.crop(labels[exp], i, j, h, w)

        # resize to the defined shape
        image = transforms_f.resize(image, self.resize_size, transforms_f.InterpolationMode.BICUBIC)
        if labels is not None:
            for exp in labels:
                labels[exp] = transforms_f.resize(labels[exp], [224, 224], transforms_f.InterpolationMode.NEAREST)

        if self.train:
            # random flipping
            if torch.rand(1) > 0.5:
                image = transforms_f.hflip(image)
                if labels is not None:
                    for exp in labels:
                        labels[exp] = transforms_f.hflip(labels[exp])

            # random augmentation
            image, labels = self.randaugment(image, labels)

        # transform to tensor
        image = transforms_f.to_tensor(image)
        if labels is not None:
            for exp in labels:
                if exp in ['depth', 'normal', 'edge']:
                    labels[exp] = transforms_f.to_tensor(labels[exp])
                else:
                    labels[exp] = (transforms_f.to_tensor(labels[exp]) * 255).long()

        # apply normalisation:
        image = transforms_f.normalize(image, mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
        if labels is not None:
            return {'rgb': image, **labels}
        else:
            return{'rgb': image}


def get_expert_labels(data_path, label_path, image_path, dataset, experts):
    image_full_path = os.path.join(data_path, dataset, image_path)
    image = Image.open(image_full_path).convert('RGB')
    if experts != 'none':
        labels = {}
        labels_info = {}
        ps = image_path.split('.')[-1]
        for exp in experts:
            if exp in ['seg_coco', 'seg_ade', 'edge', 'depth']:
                label_full_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.png'))
                if os.stat(label_full_path).st_size > 0:
                    labels[exp] = Image.open(label_full_path).convert('L')
                else:
                    labels[exp] = Image.fromarray(np.zeros([image.size[1], image.size[0]])).convert('L')
            elif exp == 'normal':
                label_full_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.png'))
                if os.stat(label_full_path).st_size > 0:
                    labels[exp] = Image.open(label_full_path).convert('RGB')
                else:
                    labels[exp] = Image.fromarray(np.zeros([image.size[1], image.size[0], 3])).convert('RGB')
            elif exp == 'obj_detection':
                label_full_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.png'))
                if os.stat(label_full_path).st_size > 0:
                    labels[exp] = Image.open(label_full_path).convert('L')
                else:
                    labels[exp] = Image.fromarray(255 * np.ones([image.size[1], image.size[0]])).convert('L')
                label_info_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.json'))
                labels_info[exp] = json.load(open(label_info_path, 'r'))
            elif exp == 'ocr_detection':
                label_full_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.png'))
                label_info_path = os.path.join(label_path, exp, dataset, image_path.replace(f'.{ps}', '.pt'))
                if os.path.exists(label_info_path):
                    labels[exp] = Image.open(label_full_path).convert('L')
                    labels_info[exp] = torch.load(label_info_path)
                else:
                    labels[exp] = Image.fromarray(255 * np.ones([image.size[1], image.size[0]])).convert('L')
                    labels_info[exp] = None

    else:
        labels, labels_info = None, None
    return image, labels, labels_info


def post_label_process(inputs, labels_info):
    eps = 1e-6
    for exp in inputs:
        if exp in ['depth', 'normal', 'edge']:  # remap to -1 to 1 range
            inputs[exp] = 2 * (inputs[exp] - inputs[exp].min()) / (inputs[exp].max() - inputs[exp].min() + eps) - 1
        
        elif exp == 'seg_coco':  # in-paint with CLIP features
            text_emb = torch.empty([64, *inputs[exp].shape[1:]])
            for l in inputs[exp].unique():
                if l == 255:
                    text_emb[:, (inputs[exp][0] == l)] = BACKGROUND_FEATURES.unsqueeze(-1)
                else:
                    text_emb[:, (inputs[exp][0] == l)] = COCO_FEATURES[l].unsqueeze(-1)
            inputs[exp] = text_emb

        elif exp == 'seg_ade':  # in-paint with CLIP features
            text_emb = torch.empty([64, *inputs[exp].shape[1:]])
            for l in inputs[exp].unique():
                if l == 255:
                    text_emb[:, (inputs[exp][0] == l)] = BACKGROUND_FEATURES.unsqueeze(-1)
                else:
                    text_emb[:, (inputs[exp][0] == l)] = ADE_FEATURES[l].unsqueeze(-1)
            inputs[exp] = text_emb

        elif exp == 'obj_detection':  # in-paint with CLIP features
            text_emb = torch.empty([64, *inputs[exp].shape[1:]])
            label_map = labels_info[exp]
            for l in inputs[exp].unique():
                if l == 255:
                    text_emb[:, (inputs[exp][0] == l)] = BACKGROUND_FEATURES.unsqueeze(-1)
                else:
                    text_emb[:, (inputs[exp][0] == l)] = DETECTION_FEATURES[label_map[str(l.item())]].unsqueeze(-1)
            inputs[exp] = {'label': text_emb, 'instance': inputs[exp]}

        elif exp == 'ocr_detection':  # in-paint with CLIP features
            text_emb = torch.empty([64, *inputs[exp].shape[1:]])
            label_map = labels_info[exp]
            for l in inputs[exp].unique():
                if l == 255:
                    text_emb[:, (inputs[exp][0] == l)] = BACKGROUND_FEATURES.unsqueeze(-1)
                else:
                    text_emb[:, (inputs[exp][0] == l)] = label_map[l.item()]['features'].unsqueeze(-1)
            inputs[exp] = text_emb
    return inputs


def pre_caption(caption, max_words=50):
    caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.capitalize())  # remove special characters
    caption = re.sub(r"\s{2,}", ' ', caption)  # remove two white spaces

    caption = caption.rstrip('\n')  # remove \num_ans_per_q symbol
    caption = caption.strip(' ')    # remove leading and trailing white spaces

    # truncate caption to the max words
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


def pre_question(question, max_words=50):
    question = re.sub(r"([.!\"()*#:;~])", ' ', question.capitalize())  # remove special characters
    question = question.strip()

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_words:
        question = ' '.join(question_words[:max_words])
    if question[-1] != '?':
        question += '?'
    return question

