# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np
import pyclipper
from shapely.geometry import Polygon


def nms(boxes, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    new_boxes = np.zeros_like(boxes)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
             for b in boxes]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    order = boxes[:, 8].argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])))
                    except:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] - minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick, new_boxes


def nms_with_char_cls(boxes, char_scores, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    new_boxes = np.zeros_like(boxes)
    new_char_scores = np.zeros_like(char_scores)
    pick = []
    suppressed = [False for _ in range(boxes.shape[0])]
    areas = [Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])]).area
             for b in boxes]
    polygons = pyclipper.scale_to_clipper(boxes[:, :8].reshape((-1, 4, 2)))
    order = boxes[:, 8].argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])))
                    except:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) >= num_neig:
                neighbours.append(i)
                temp_scores = (boxes[neighbours, 8] - minScore).reshape((-1, 1))
                new_boxes[i, :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
                new_boxes[i, 8] = boxes[i, 8]
                new_char_scores[i, :] = (char_scores[neighbours, :] * temp_scores).sum(axis=0) / temp_scores.sum()
            else:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick, new_boxes, new_char_scores


def softnms(boxes, box_scores, char_scores=None, overlapThresh=0.3,
                          threshold=0.8, neighbourThresh=0.5, num_neig=0):
    scores = box_scores.copy()
    new_boxes = boxes[:, 0: 8].copy()
    if char_scores is not None:
        new_char_scores = char_scores.copy()
    polygons = [pyclipper.scale_to_clipper(poly.reshape((-1, 2))) for poly in new_boxes]
    areas = [pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(
             pyclipper.Area(poly))) for poly in polygons]
    areas = [abs(_) for _ in areas]
    N = boxes.shape[0]
    order = np.arange(N)
    i = 0
    while i < N:
        max_pos = scores[order[i: N]].argmax() + i
        order[i], order[max_pos] = order[max_pos], order[i]
        pos = i + 1
        neighbours = list()
        while pos < N:
            try:
                pc = pyclipper.Pyclipper()
                pc.AddPath(polygons[order[i]], pyclipper.PT_CLIP, True)
                pc.AddPaths([polygons[order[pos]]], pyclipper.PT_SUBJECT, True)
                solution = pc.Execute(pyclipper.CT_INTERSECTION)
                if len(solution) == 0:
                    inter = 0
                else:
                    inter = pyclipper.scale_from_clipper(
                        pyclipper.scale_from_clipper(
                            pyclipper.Area(solution[0])))
            except Exception:
                inter = 0
            union = areas[order[i]] + areas[order[pos]] - inter
            iou = inter / union if union > 0 else 0
            if iou > neighbourThresh:
                neighbours.append(order[pos])
            weight = np.exp(-(iou **2) / 0.5)
            scores[order[pos]] *= weight
            if scores[order[pos]] < threshold:
                order[pos], order[N - 1] = order[N - 1], order[pos]
                N -= 1
                pos -= 1
            pos += 1
        if len(neighbours) >= num_neig:
            neighbours.append(order[i])
            temp_scores = box_scores[neighbours].reshape((-1, 1))
            new_boxes[order[i], :8] = (boxes[neighbours, :8] * temp_scores).sum(axis=0) / temp_scores.sum()
            if char_scores is not None:
                new_char_scores[order[i], :] = (char_scores[neighbours, :] * temp_scores).sum(axis=0) / temp_scores.sum()
        else:
            order[i], order[N - 1] = order[N - 1], order[i]
            N -= 1
            i -= 1
        i += 1
    keep = [order[_] for _ in range(N)]
    if char_scores is not None:
        return keep, new_boxes, new_char_scores
    else:
        return keep, new_boxes


def nms_poly(polys, scores, overlapThresh, neighbourThresh=0.5, minScore=0, num_neig=0):
    pick = list()
    suppressed = [False for _ in range(len(polys))]
    polygons = [pyclipper.scale_to_clipper(poly.reshape((-1, 2))) for poly in polys]
    areas = [pyclipper.scale_from_clipper(pyclipper.scale_from_clipper(
             pyclipper.Area(poly))) for poly in polygons]
    areas = [abs(_) for _ in areas]
    order = np.array(scores).argsort()[::-1]
    for _i, i in enumerate(order):
        if suppressed[i] is False:
            pick.append(i)
            neighbours = list()
            for j in order[_i+1:]:
                if suppressed[j] is False:
                    try:
                        pc = pyclipper.Pyclipper()
                        pc.AddPath(polygons[i], pyclipper.PT_CLIP, True)
                        pc.AddPaths([polygons[j]], pyclipper.PT_SUBJECT, True)
                        solution = pc.Execute(pyclipper.CT_INTERSECTION)
                        if len(solution) == 0:
                            inter = 0
                        else:
                            inter = pyclipper.scale_from_clipper(
                                pyclipper.scale_from_clipper(
                                    pyclipper.Area(solution[0])))
                    except Exception as e:
                        inter = 0
                    union = areas[i] + areas[j] - inter
                    iou = inter / union if union > 0 else 0
                    if union > 0 and iou > overlapThresh:
                        suppressed[j] = True
                    if iou > neighbourThresh:
                        neighbours.append(j)
            if len(neighbours) < num_neig:
                for ni in neighbours:
                    suppressed[ni] = False
                pick.pop()
    return pick
