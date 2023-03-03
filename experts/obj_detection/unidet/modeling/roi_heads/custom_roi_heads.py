import numpy as np
import json
import math
import torch
from torch import nn
from torch.autograd.function import Function

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient, CascadeROIHeads
from detectron2.modeling.roi_heads.box_head import FastRCNNConvFCHead, ROI_BOX_HEAD_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .custom_fast_rcnn import CustomFastRCNNOutputLayers

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = CustomFastRCNNOutputLayers(cfg, ret['box_head'].output_shape)
        return ret

@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
        ret['box_predictors'] = box_predictors
        return ret

