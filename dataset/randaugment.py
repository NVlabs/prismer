# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import random
import numpy as np
import torch

from PIL import Image, ImageOps, ImageEnhance, ImageDraw


fillmask = {'depth': 0, 'normal': 0, 'edge': 0, 'seg_coco': 255, 'seg_ade': 255,
            'obj_detection': 255, 'ocr_detection': 255}
fillcolor = (0, 0, 0)


def affine_transform(pair, affine_params):
    img, label = pair
    img = img.transform(img.size, Image.AFFINE, affine_params,
                        resample=Image.BILINEAR, fillcolor=fillcolor)
    if label is not None:
        for exp in label:
            label[exp] = label[exp].transform(label[exp].size, Image.AFFINE, affine_params,
                                              resample=Image.NEAREST, fillcolor=fillmask[exp])
    return img, label


def ShearX(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0))


def ShearY(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, v, 1, 0))


def TranslateX(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[0]
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[1]
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    img, label = pair
    img = img.rotate(v, fillcolor=fillcolor)
    if label is not None:
        for exp in label:
            label[exp] = label[exp].rotate(v, resample=Image.NEAREST, fillcolor=fillmask[exp])
    return img, label


def AutoContrast(pair, _):
    img, label = pair
    return ImageOps.autocontrast(img), label


def Invert(pair, _):
    img, label = pair
    return ImageOps.invert(img), label


def Equalize(pair, _):
    img, label = pair
    return ImageOps.equalize(img), label


def Flip(pair, _):  # not from the paper
    img, label = pair
    return ImageOps.mirror(img), ImageOps.mirror(label)


def Solarize(pair, v):  # [0, 256]
    img, label = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), label


def Posterize(pair, v):  # [4, 8]
    img, label = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), label


def Posterize2(pair, v):  # [0, 4]
    img, label = pair
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v), label


def Contrast(pair, v):  # [0.1,1.9]
    img, label = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), label


def Color(pair, v):  # [0.1,1.9]
    img, label = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v), label


def Brightness(pair, v):  # [0.1,1.9]
    img, label = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), label


def Sharpness(pair, v):  # [0.1,1.9]
    img, label = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), label


def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    img, label = pair
    v = v * img.size[0]
    return CutoutAbs(img, v), label


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Identity(pair, v):
    return pair


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0),
        (ShearX, 0., 0.3),  # 0
        (ShearY, 0., 0.3),  # 1
        (TranslateX, 0., 0.33),  # 2
        (TranslateY, 0., 0.33),  # 3
        (Rotate, 0, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        # (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        # (Solarize, 0, 110),  # 8
        # (Posterize, 4, 8),  # 9
        # (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 10]
        self.augment_list = augment_list()

    def __call__(self, img, label):
        pair = img, label
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            pair = op(pair, val)
        return pair
