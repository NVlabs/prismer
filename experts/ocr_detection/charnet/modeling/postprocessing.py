# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
import numpy as np
import cv2
import editdistance
from experts.ocr_detection.charnet.modeling.utils import rotate_rect
from experts.ocr_detection.charnet.modeling.rotated_nms import nms, nms_with_char_cls, softnms, nms_poly
from shapely.geometry import Polygon
import pyclipper


def load_lexicon(path):
    lexicon = list()
    with open(path, 'rt') as fr:
        for line in fr:
            if line.startswith('#'):
                pass
            else:
                lexicon.append(line.strip())
    return lexicon


def load_char_dict(path, seperator=chr(31)):
    char_dict = dict()
    with open(path, 'rt') as fr:
        for line in fr:
            sp = line.strip('\n').split(seperator)
            char_dict[int(sp[1])] = sp[0].upper()
    return char_dict


class WordInstance:
    def __init__(self, word_bbox, word_bbox_score, text, text_score, char_scores):
        self.word_bbox = word_bbox
        self.word_bbox_score = word_bbox_score
        self.text = text
        self.text_score = text_score
        self.char_scores = char_scores


class OrientedTextPostProcessing(nn.Module):
    def __init__(
            self, word_min_score, word_stride,
            word_nms_iou_thresh, char_stride,
            char_min_score, num_char_class,
            char_nms_iou_thresh, char_dict_file,
            word_lexicon_path
    ):
        super(OrientedTextPostProcessing, self).__init__()
        self.word_min_score = word_min_score
        self.word_stride = word_stride
        self.word_nms_iou_thresh = word_nms_iou_thresh
        self.char_stride = char_stride
        self.char_min_score = char_min_score
        self.num_char_class = num_char_class
        self.char_nms_iou_thresh = char_nms_iou_thresh
        self.char_dict = load_char_dict(char_dict_file)
        self.lexicon = load_lexicon(word_lexicon_path)

    def forward(
            self, pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            im_scale_w, im_scale_h,
            original_im_w, original_im_h
    ):
        ss_word_bboxes = self.parse_word_bboxes(
            pred_word_fg, pred_word_tblr, pred_word_orient,
            im_scale_w, im_scale_h, original_im_w, original_im_h
        )
        char_bboxes, char_scores = self.parse_char(
            pred_word_fg, pred_char_fg, pred_char_tblr, pred_char_cls,
            im_scale_w, im_scale_h, original_im_w, original_im_h
        )
        word_instances = self.parse_words(
            ss_word_bboxes, char_bboxes,
            char_scores, self.char_dict
        )

        word_instances = self.filter_word_instances(word_instances, self.lexicon)

        return char_bboxes, char_scores, word_instances

    def parse_word_bboxes(
            self, pred_word_fg, pred_word_tblr,
            pred_word_orient, scale_w, scale_h,
            W, H
    ):
        word_stride = self.word_stride
        word_keep_rows, word_keep_cols = np.where(pred_word_fg > self.word_min_score)
        oriented_word_bboxes = np.zeros((word_keep_rows.shape[0], 9), dtype=np.float32)
        for idx in range(oriented_word_bboxes.shape[0]):
            y, x = word_keep_rows[idx], word_keep_cols[idx]
            t, b, l, r = pred_word_tblr[:, y, x]
            o = pred_word_orient[y, x]
            score = pred_word_fg[y, x]
            four_points = rotate_rect(
                scale_w * word_stride * (x-l), scale_h * word_stride * (y-t),
                scale_w * word_stride * (x+r), scale_h * word_stride * (y+b),
                o, scale_w * word_stride * x, scale_h * word_stride * y)
            oriented_word_bboxes[idx, :8] = np.array(four_points, dtype=np.float32).flat
            oriented_word_bboxes[idx, 8] = score
        keep, oriented_word_bboxes = nms(oriented_word_bboxes, self.word_nms_iou_thresh, num_neig=1)
        oriented_word_bboxes = oriented_word_bboxes[keep]
        oriented_word_bboxes[:, :8] = oriented_word_bboxes[:, :8].round()
        oriented_word_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(W-1, oriented_word_bboxes[:, 0:8:2]))
        oriented_word_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(H-1, oriented_word_bboxes[:, 1:8:2]))
        return oriented_word_bboxes

    def parse_char(
            self, pred_word_fg, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            scale_w, scale_h, W, H
    ):
        char_stride = self.char_stride
        if pred_word_fg.shape == pred_char_fg.shape:
            char_keep_rows, char_keep_cols = np.where(
                (pred_word_fg > self.word_min_score) & (pred_char_fg > self.char_min_score))
        else:
            th, tw = pred_char_fg.shape
            word_fg_mask = cv2.resize((pred_word_fg > self.word_min_score).astype(np.uint8),
                                      (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.bool)
            char_keep_rows, char_keep_cols = np.where(
                word_fg_mask & (pred_char_fg > self.char_min_score))

        oriented_char_bboxes = np.zeros((char_keep_rows.shape[0], 9), dtype=np.float32)
        char_scores = np.zeros((char_keep_rows.shape[0], self.num_char_class), dtype=np.float32)
        for idx in range(oriented_char_bboxes.shape[0]):
            y, x = char_keep_rows[idx], char_keep_cols[idx]
            t, b, l, r = pred_char_tblr[:, y, x]
            o = 0.0  # pred_char_orient[y, x]
            score = pred_char_fg[y, x]
            four_points = rotate_rect(
                scale_w * char_stride * (x-l), scale_h * char_stride * (y-t),
                scale_w * char_stride * (x+r), scale_h * char_stride * (y+b),
                o, scale_w * char_stride * x, scale_h * char_stride * y)
            oriented_char_bboxes[idx, :8] = np.array(four_points, dtype=np.float32).flat
            oriented_char_bboxes[idx, 8] = score
            char_scores[idx, :] = pred_char_cls[:, y, x]
        keep, oriented_char_bboxes, char_scores = nms_with_char_cls(
            oriented_char_bboxes, char_scores, self.char_nms_iou_thresh, num_neig=1
        )
        oriented_char_bboxes = oriented_char_bboxes[keep]
        oriented_char_bboxes[:, :8] = oriented_char_bboxes[:, :8].round()
        oriented_char_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(W-1, oriented_char_bboxes[:, 0:8:2]))
        oriented_char_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(H-1, oriented_char_bboxes[:, 1:8:2]))
        char_scores = char_scores[keep]
        return oriented_char_bboxes, char_scores

    def filter_word_instances(self, word_instances, lexicon):
        def match_lexicon(text, lexicon):
            min_dist, min_idx = 1e8, None
            for idx, voc in enumerate(lexicon):
                dist = editdistance.eval(text.upper(), voc.upper())
                if dist == 0:
                    return 0, text
                else:
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = idx
            return min_dist, lexicon[min_idx]

        def filter_and_correct(word_ins, lexicon):
            if word_ins.text_score >= 0.80:
                if not word_ins.text.isalpha() and word_ins.text_score >= 0.9:
                    return word_ins
                elif word_ins.text_score >= 0.98:
                    return word_ins
                else:
                    dist, voc = match_lexicon(word_ins.text, lexicon)
                    word_ins.text = voc
                    word_ins.text_edst = dist

                    if len(voc) <= 2:
                        min_dist = 0
                    elif len(voc) <= 5:
                        min_dist = 1
                    else:
                        min_dist = 2

                    if dist <= min_dist:
                        return word_ins
                    else:
                        return None
            else:
                return None
        valid_word_instances = list()
        for word_ins in word_instances:
            word_ins = filter_and_correct(word_ins, lexicon)
            if word_ins is not None:
                valid_word_instances.append(word_ins)
        return valid_word_instances

    def nms_word_instances(self, word_instances, h, w, edst=False):
        word_bboxes = np.zeros((len(word_instances), 9), dtype=np.float32)
        for idx, word_ins in enumerate(word_instances):
            word_bboxes[idx, :8] = word_ins.word_bbox
            word_bboxes[idx, 8] = word_ins.word_bbox_score * 1 + word_ins.text_score
            if edst is True:
                text_edst = getattr(word_ins, 'text_edst', 0)
                word_bboxes[idx, 8] -= (word_ins.text_score / len(word_ins.text)) * text_edst
        keep, word_bboxes = nms(word_bboxes, self.word_nms_iou_thresh, num_neig=0)
        word_bboxes = word_bboxes[keep]
        word_bboxes[:, :8] = word_bboxes[:, :8].round()
        word_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(w-1, word_bboxes[:, 0:8:2]))
        word_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(h-1, word_bboxes[:, 1:8:2]))
        word_instances = [word_instances[idx] for idx in keep]
        for word_ins, word_bbox, in zip(word_instances, word_bboxes):
            word_ins.word_bbox[:8] = word_bbox[:8]
        return word_instances

    def parse_words(self, word_bboxes, char_bboxes, char_scores, char_dict):
        def match(word_bbox, word_poly, char_bbox, char_poly):
            word_xs = word_bbox[0:8:2]
            word_ys = word_bbox[1:8:2]
            char_xs = char_bbox[0:8:2]
            char_ys = char_bbox[1:8:2]
            if char_xs.min() > word_xs.max() or\
               char_xs.max() < word_xs.min() or\
               char_ys.min() > word_ys.max() or\
               char_ys.max() < word_ys.min():
                return 0
            else:
                inter = char_poly.intersection(word_poly)
                return inter.area / (char_poly.area + word_poly.area - inter.area)

        def decode(char_scores):
            max_indices = char_scores.argmax(axis=1)
            text = [char_dict[idx] for idx in max_indices]
            scores = [char_scores[idx, max_indices[idx]] for idx in range(max_indices.shape[0])]
            return ''.join(text), np.array(scores, dtype=np.float32).mean()

        def recog(word_bbox, char_bboxes, char_scores):
            word_vec = np.array([1, 0], dtype=np.float32)
            char_vecs = (char_bboxes.reshape((-1, 4, 2)) - word_bbox[0:2]).mean(axis=1)
            proj = char_vecs.dot(word_vec)
            order = np.argsort(proj)
            text, score = decode(char_scores[order])
            return text, score, char_scores[order]

        word_bbox_scores = word_bboxes[:, 8]
        char_bbox_scores = char_bboxes[:, 8]
        word_bboxes = word_bboxes[:, :8]
        char_bboxes = char_bboxes[:, :8]
        word_polys = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
                      for b in word_bboxes]
        char_polys = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
                      for b in char_bboxes]
        num_word = word_bboxes.shape[0]
        num_char = char_bboxes.shape[0]
        word_instances = list()
        word_chars = [list() for _ in range(num_word)]

        if num_word == 0:
            return word_instances

        for idx in range(num_char):
            char_bbox = char_bboxes[idx]
            char_poly = char_polys[idx]
            match_scores = np.zeros((num_word,), dtype=np.float32)
            for jdx in range(num_word):
                word_bbox = word_bboxes[jdx]
                word_poly = word_polys[jdx]
                match_scores[jdx] = match(word_bbox, word_poly, char_bbox, char_poly)
            jdx = np.argmax(match_scores)
            if match_scores[jdx] > 0:
                word_chars[jdx].append(idx)

        for idx in range(num_word):
            char_indices = word_chars[idx]
            if len(char_indices) > 0:
                text, text_score, tmp_char_scores = recog(
                    word_bboxes[idx],
                    char_bboxes[char_indices],
                    char_scores[char_indices]
                )
                word_instances.append(WordInstance(
                    word_bboxes[idx],
                    word_bbox_scores[idx],
                    text, text_score,
                    tmp_char_scores
                ))
        return word_instances
