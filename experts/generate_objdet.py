# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import os
import json
import copy
import PIL.Image as Image
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
    
from experts.model_bank import load_expert_model
from experts.obj_detection.generate_dataset import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm

model, transform = load_expert_model(task='obj_detection')
accelerator = Accelerator(mixed_precision='fp16')

config = yaml.load(open('configs/experts.yaml', 'r'), Loader=yaml.Loader)
data_path = config['data_path']
save_path = config['save_path']

depth_path = os.path.join(save_path, 'depth', data_path.split('/')[-1])
batch_size = 32
dataset = Dataset(data_path, depth_path, transform)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

model, data_loader = accelerator.prepare(model, data_loader)


def get_mask_labels(depth, instance_boxes, instance_id):
    obj_masks = []
    obj_ids = []
    for i in range(len(instance_boxes)):
        is_duplicate = False
        mask = torch.zeros_like(depth)
        x1, y1, x2, y2 = instance_boxes[i][0].item(), instance_boxes[i][1].item(), \
                         instance_boxes[i][2].item(), instance_boxes[i][3].item()
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        for j in range(len(obj_masks)):
            if ((mask + obj_masks[j]) == 2).sum() / ((mask + obj_masks[j]) > 0).sum() > 0.95:
                is_duplicate = True
                break
        if not is_duplicate:
            obj_masks.append(mask)
            obj_ids.append(instance_id[i])

    obj_masked_modified = copy.deepcopy(obj_masks[:])
    for i in range(len(obj_masks) - 1):
        mask1 = obj_masks[i]
        mask1_ = obj_masked_modified[i]
        for j in range(i + 1, len(obj_masks)):
            mask2 = obj_masks[j]
            mask2_ = obj_masked_modified[j]
            # case 1: if they don't intersect we don't touch them
            if ((mask1 + mask2) == 2).sum() == 0:
                continue
            # case 2: the entire object 1 is inside of object 2, we say object 1 is in front of object 2:
            elif (((mask1 + mask2) == 2).float() - mask1).sum() == 0:
                mask2_ -= mask1_
            # case 3: the entire object 2 is inside of object 1, we say object 2 is in front of object 1:
            elif (((mask1 + mask2) == 2).float() - mask2).sum() == 0:
                mask1_ -= mask2_
            # case 4: use depth to check object order:
            else:
                # object 1 is closer
                if (depth * mask1).sum() / mask1.sum() > (depth * mask2).sum() / mask2.sum():
                    mask2_ -= ((mask1 + mask2) == 2).float()
                # object 2 is closer
                if (depth * mask1).sum() / mask1.sum() < (depth * mask2).sum() / mask2.sum():
                    mask1_ -= ((mask1 + mask2) == 2).float()

    final_mask = torch.ones_like(depth) * 255
    instance_labels = {}
    for i in range(len(obj_masked_modified)):
        final_mask = final_mask.masked_fill(obj_masked_modified[i] > 0, i)
        instance_labels[i] = obj_ids[i].item()
    return final_mask, instance_labels


with torch.no_grad():
    for i, test_data in enumerate(tqdm(data_loader)):
        test_pred = model(test_data)
        for k in range(len(test_pred)):
            instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor
            instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
            depth = test_data[k]['depth']

            final_mask, instance_labels = get_mask_labels(depth, instance_boxes, instance_id)

            img_path_split = test_data[k]['image_path'].split('/')
            im_save_path = os.path.join(save_path, 'obj_detection', img_path_split[-3], img_path_split[-2])
            ps = test_data[k]['image_path'].split('.')[-1]
            os.makedirs(im_save_path, exist_ok=True)

            height, width = test_data[k]['true_height'], test_data[k]['true_width']
            final_mask = Image.fromarray(final_mask.cpu().numpy()).convert('L')
            final_mask = final_mask.resize((height, width), resample=Image.Resampling.NEAREST)
            final_mask.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))

            with open(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.json')), 'w') as fp:
                json.dump(instance_labels, fp)
