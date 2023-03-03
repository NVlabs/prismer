import logging
import math
import json
import os
from typing import Dict, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

__all__ = ["CustomFastRCNNOutputLayers"]


def _load_class_freq(cfg):
    freq_weight = None
    if cfg.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS or cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS:
        # print('Loading', cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        if not os.path.exists(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH):
            return
        cat_info = json.load(open(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH, 'r'))
        cat_info = torch.tensor(
            [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])],
            device=torch.device(cfg.MODEL.DEVICE))
        if cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS and \
            cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT > 0.:
            freq_weight = \
                cat_info.float() ** cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT
        else:
            thresh, _ = torch.kthvalue(
                cat_info,
                len(cat_info) - cfg.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT + 1)
            freq_weight = (cat_info < thresh.item()).float()

    return freq_weight


def _load_class_hierarchy(cfg):
    hierarchy_weight = None
    if cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_IGNORE:
        if not os.path.exists(cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH):
            return
        # print('Loading', cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH)
        hierarchy_data = json.load(
            open(cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH, 'r'))
        parents = {int(k): v for k, v in hierarchy_data['parents'].items()}
        chirlds = {int(k): v for k, v in hierarchy_data['childs'].items()}
        categories = hierarchy_data['categories']
        continousid = sorted([x['id'] for x in categories])
        catid2continous = {x['id']: continousid.index(x['id']) \
            for x in categories}
        C = len(categories)
        is_parents = torch.zeros((C + 1, C), device=torch.device(cfg.MODEL.DEVICE)).float()
        is_chirlds = torch.zeros((C + 1, C), device=torch.device(cfg.MODEL.DEVICE)).float()
        for c in categories:
            cat_id = catid2continous[c['id']]
            is_parents[cat_id, [catid2continous[x] for x in parents[c['id']]]] = 1
            is_chirlds[cat_id, [catid2continous[x] for x in chirlds[c['id']]]] = 1
        assert (is_parents * is_chirlds).sum() == 0
        if cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_POS_PARENTS:
            hierarchy_weight = (1 - is_chirlds, is_parents[:C])
        else:
            hierarchy_weight = 1 - (is_parents + is_chirlds) # (C + 1) x C

    return hierarchy_weight


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(
        self, 
        cfg, 
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(cfg, input_shape, **kwargs)
        self.use_sigmoid_ce = cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
        self.use_eql_loss = cfg.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS
        self.use_fed_loss = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        self.fed_loss_num_cat = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT
        self.pos_parents = cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_POS_PARENTS
        self.hierarchy_ignore = cfg.MODEL.ROI_BOX_HEAD.HIERARCHY_IGNORE

        if self.use_sigmoid_ce:
            prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        self.freq_weight = _load_class_freq(cfg)
        hierarchy_weight = _load_class_hierarchy(cfg)
        if self.pos_parents and (hierarchy_weight is not None):
            self.hierarchy_weight = hierarchy_weight[0] # (C + 1) x C
            self.is_parents = hierarchy_weight[1]
        else:
            self.hierarchy_weight = hierarchy_weight # (C + 1) x C


    def predict_probs(self, predictions, proposals):
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)

        return probs.split(num_inst_per_image, dim=0)


    def sigmoid_cross_entropy_loss(
        self, pred_class_logits, gt_classes, use_advanced_loss=True):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = self.pred_class_logits.shape[0]
        C = self.pred_class_logits.shape[1] - 1

        target = self.pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
        if use_advanced_loss and (self.freq_weight is not None) and \
            self.use_fed_loss: # fedloss
            appeared = torch.unique(gt_classes) # C'
            prob = appeared.new_ones(C + 1).float()
            if len(appeared) < self.fed_loss_num_cat:
                if self.fed_loss_freq_weight > 0:
                    prob[:C] = self.freq_weight.float().clone()
                else:
                    prob[:C] = prob[:C] * (1 - self.freq_weight)
                prob[appeared] = 0
                more_appeared = torch.multinomial(
                    prob, self.fed_loss_num_cat - len(appeared),
                    replacement=False)
                appeared = torch.cat([appeared, more_appeared])
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w

        if use_advanced_loss and (self.hierarchy_weight is not None) and \
            self.hierarchy_ignore:
            if self.pos_parents:
                target = torch.mm(target, self.is_parents) + target # B x C
            hierarchy_w = self.hierarchy_weight[gt_classes] # B x C
            weight = weight * hierarchy_w

        cls_loss = F.binary_cross_entropy_with_logits(
            self.pred_class_logits[:, :-1], target, reduction='none') # B x C
        return torch.sum(cls_loss * weight) / B


    def losses(self, predictions, proposals, use_advanced_loss=True):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)


        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)


        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(
                scores, gt_classes, use_advanced_loss)
        else:
            assert not use_advanced_loss
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)
        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        }