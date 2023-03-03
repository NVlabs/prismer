# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

from torch.utils.data import Dataset
from dataset.utils import *


class VQA(Dataset):
    def __init__(self, config, train=True):
        self.data_path = config['data_path']
        self.label_path = config['label_path']
        self.experts = config['experts']
        self.transform = Transform(resize_resolution=config['image_resolution'], scale_size=[0.5, 1.0], train=train)
        self.train = train

        if train:
            self.data_list = []
            if 'vqav2' in config['datasets']:
                self.data_list += json.load(open(os.path.join(self.data_path, 'vqav2_train_val.json'), 'r'))
            if 'vg' in config['datasets']:
                self.data_list += json.load(open(os.path.join(self.data_path, 'vg_qa.json'), 'r'))
        else:
            self.data_list = json.load(open(os.path.join(self.data_path, 'vqav2_test.json'), 'r'))
            self.answer_list = json.load(open(os.path.join(self.data_path, 'answer_list.json'), 'r'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        if data['dataset'] == 'vqa':
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, data['image'], 'vqav2', self.experts)
        elif data['dataset'] == 'vg':
            image, labels, labels_info = get_expert_labels(self.data_path, self.label_path, data['image'], 'vg', self.experts)

        experts = self.transform(image, labels)
        experts = post_label_process(experts, labels_info)

        if self.train:
            question = pre_question(data['question'], max_words=30)
            answers = data['answer']
            weights = torch.tensor(data['weight']) if data['dataset'] != 'vg' else torch.tensor(0.2)
            return experts, question, answers, weights
        else:
            question = pre_question(data['question'], max_words=30)
            question_id = data['question_id']
            return experts, index, question, question_id
