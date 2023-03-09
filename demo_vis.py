import glob
import os
import json
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

from utils import create_ade20k_label_colormap

obj_label_map = torch.load('dataset/detection_features.pt')['labels']
coco_label_map = torch.load('dataset/coco_features.pt')['labels']
ade_color = create_ade20k_label_colormap()

file_path = 'helpers/images'
expert_path = 'helpers/labels'
plt.ioff()


def get_label_path(file_name, expert_name, with_suffix=False):
    file_suffix = '.png' if not with_suffix else '_.png'
    label_name = ''.join(file_name.split('.')[:-1] + [file_suffix])
    label_path = os.path.join(expert_path, expert_name, label_name)
    return label_path


def depth_prettify(file_name):
    label_path = get_label_path(file_name, 'depth')
    save_path = get_label_path(file_name, 'depth', True)
    depth = plt.imread(label_path)
    plt.imsave(save_path, depth, cmap='rainbow')


def obj_detection_prettify(file_name):
    label_path = get_label_path(file_name, 'obj_detection')
    save_path = get_label_path(file_name, 'obj_detection', True)

    rgb = plt.imread(file_name)
    obj_labels = plt.imread(label_path)
    obj_labels_dict = json.load(open(label_path.replace('.png', '.json')))

    plt.imshow(rgb)

    num_objs = np.unique(obj_labels)[:-1].max()
    plt.imshow(obj_labels, cmap='terrain', vmax=num_objs + 1 / 255., alpha=0.5)

    for i in np.unique(obj_labels)[:-1]:
        obj_idx_all = np.where(obj_labels == i)
        obj_idx = random.randint(0, len(obj_idx_all[0]))
        x, y = obj_idx_all[1][obj_idx], obj_idx_all[0][obj_idx]
        obj_name = obj_label_map[obj_labels_dict[str(int(i * 255))]]
        plt.text(x, y, obj_name, c='white', horizontalalignment='center', verticalalignment='center')

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()


def seg_prettify(file_name):
    label_path = get_label_path(file_name, 'seg_coco')
    save_path = get_label_path(file_name, 'seg_coco', True)

    rgb = plt.imread(file_name)
    seg_labels = plt.imread(label_path)

    plt.imshow(rgb)

    seg_map = np.zeros(list(seg_labels.shape) + [3], dtype=np.int16)
    for i in np.unique(seg_labels):
        seg_map[seg_labels == i] = ade_color[int(i * 255)]

    plt.imshow(seg_map, alpha=0.5)

    for i in np.unique(seg_labels):
        obj_idx_all = np.where(seg_labels == i)
        obj_idx = random.randint(0, len(obj_idx_all[0]))
        x, y = obj_idx_all[1][obj_idx], obj_idx_all[0][obj_idx]
        obj_name = coco_label_map[int(i * 255)]
        plt.text(x, y, obj_name, c='white', horizontalalignment='center', verticalalignment='center')

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()


def ocr_detection_prettify(file_name):
    label_path = get_label_path(file_name, 'ocr_detection')
    save_path = get_label_path(file_name, 'ocr_detection', True)

    if os.path.exists(label_path):
        rgb = plt.imread(file_name)
        ocr_labels = plt.imread(label_path)
        ocr_labels_dict = torch.load(label_path.replace('.png', '.pt'))

        plt.imshow(rgb)
        plt.imshow((1 - ocr_labels) < 1, cmap='gray', alpha=0.8)

        for i in np.unique(ocr_labels)[:-1]:
            text_idx_all = np.where(ocr_labels == i)
            x, y = text_idx_all[1].mean(), text_idx_all[0].mean()
            text = ocr_labels_dict[int(i * 255)]['text']
            plt.text(x, y, text, c='white', horizontalalignment='center', verticalalignment='center')

        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
    else:
        rgb = plt.imread(file_name)
        ocr_labels = np.ones_like(rgb, dtype=np.float32())

        plt.imshow(rgb)
        plt.imshow(ocr_labels, cmap='gray', alpha=0.8)

        x, y = rgb.shape[1] / 2, rgb.shape[0] / 2
        plt.text(x, y, 'No text detected', c='black', horizontalalignment='center', verticalalignment='center')

        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()


im_list = glob.glob(file_path + '/*.jpg') + glob.glob(file_path + '/*.png') + glob.glob(file_path + '/*.jpeg')

# prettify labels first:
for i in range(len(im_list)):
    depth_prettify(im_list[i])
    seg_prettify(im_list[i])
    ocr_detection_prettify(im_list[i])
    obj_detection_prettify(im_list[i])

pretty = {'depth': True, 'normal': False, 'edge': False,
          'obj_detection': True, 'ocr_detection': True, 'seg_coco': True}

# plot expert labels
for im_path in im_list:
    fig, axs = plt.subplots(1, 7, figsize=(20, 4))
    rgb = plt.imread(im_path)
    axs[0].imshow(rgb)
    axs[0].axis('off')
    axs[0].set_title('RGB')

    for j in range(6):
        label_name = list(pretty.keys())[j]
        label_path = get_label_path(im_path, label_name, with_suffix=pretty[label_name])
        label = plt.imread(label_path)
        if label_name != 'edge':
            axs[j + 1].imshow(label)
        else:
            axs[j + 1].imshow(label, cmap='gray')

        axs[j + 1].axis('off')
        axs[j + 1].set_title(label_name)

    caption_path = ''.join(im_path.split('.')[:-1] + ['.txt'])
    with open(caption_path) as f:
        caption = f.readlines()[0]

    plt.suptitle(caption)
    plt.tight_layout()

plt.show()
