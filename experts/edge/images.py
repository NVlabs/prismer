import cv2
import numpy as np
import torch


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def fuse_edge(pred):
    edge_maps = []
    for i in pred:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)

    fuses = []
    for idx in range(tensor.shape[1]):
        tmp = tensor[:, idx, ...]
        tmp = np.squeeze(tmp)

        # Iterate our all 7 NN outputs for a particular image
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)

            if i == 6:
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
        fuses.append(fuse)
    return fuses


