import logging
import math
from typing import Dict, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from .custom_fast_rcnn import CustomFastRCNNOutputLayers

class MultiDatasetFastRCNNOutputLayers(CustomFastRCNNOutputLayers):
    def __init__(
        self,
        cfg,
        num_classes_list,
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(cfg, input_shape, **kwargs)
        del self.cls_score
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
        if cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
        else:
            bias_value = 0
        self.openimage_index = cfg.MULTI_DATASET.DATASETS.index('oid')
        self.num_datasets = len(num_classes_list)
        self.cls_score = nn.ModuleList()
        for num_classes in num_classes_list:
            self.cls_score.append(nn.Linear(input_size, num_classes + 1))
            nn.init.normal_(self.cls_score[-1].weight, std=0.01)
            nn.init.constant_(self.cls_score[-1].bias, bias_value)

    def forward(self, x, dataset_source=-1):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        if dataset_source >= 0:
            scores = self.cls_score[dataset_source](x)
        else:
            scores = [self.cls_score[d](x) for d in range(self.num_datasets)]
        return scores, proposal_deltas

    def losses(self, predictions, proposals, dataset_source):
        use_advanced_loss = (dataset_source == self.openimage_index)
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