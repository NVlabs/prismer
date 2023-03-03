import json
import torch
from torch import nn
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient
from detectron2.modeling.box_regression import Box2BoxTransform
from .multi_dataset_fast_rcnn import MultiDatasetFastRCNNOutputLayers
from .custom_roi_heads import CustomCascadeROIHeads

from detectron2.utils.events import get_event_storage

@ROI_HEADS_REGISTRY.register()
class UnifiedCascadeROIHeads(CustomCascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        self.dataset_names = cfg.MULTI_DATASET.DATASETS
        self.unified_map_back = cfg.MODEL.ROI_BOX_HEAD.UNIFIED_MAP_BACK
        self.openimage_index = self.dataset_names.index('oid')
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        label_map = json.load(
            open(cfg.MULTI_DATASET.UNIFIED_LABEL_FILE, 'r'))['label_map']
        # add background class
        self.dataset_inds = {i: torch.tensor(
            [x for x in label_map[d]] + [num_classes]).long().to(
            torch.device(cfg.MODEL.DEVICE)) \
            for i, d in enumerate(self.dataset_names)}

        self.back_map = {}
        for i, d in enumerate(self.dataset_names):
            self.back_map[i] = self.dataset_inds[i].new_zeros(num_classes + 1)
            self.back_map[i][self.dataset_inds[i]] = \
                torch.arange(
                    len(self.dataset_inds[i]), 
                    device=torch.device(cfg.MODEL.DEVICE))

        return ret 

    def forward(self, images, features, proposals, targets=None, eval_dataset=-1):
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            dataset_sources = [target._dataset_source for target in targets]
        else:
            dataset_sources = [eval_dataset for _ in range(len(images))]
        assert len(set(dataset_sources)) == 1, dataset_sources
        dataset_source = dataset_sources[0]
        del images

        if self.training:
            losses = self._forward_box(features, proposals, targets, dataset_source)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, dataset_source=dataset_source)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def _forward_box(self, features, proposals, targets=None, dataset_source=-1):
        features = [features[f] for f in self.box_in_features]
        head_outputs = [] # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k, dataset_source)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("{}_stage{}".format(
                    self.dataset_names[dataset_source], stage)):
                    stage_losses = predictor.losses(predictions, proposals, 
                        use_advanced_loss=(dataset_source==self.openimage_index))
                losses.update({"{}_{}_stage{}".format(
                    self.dataset_names[dataset_source], 
                    k, stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    def _run_stage(self, features, proposals, stage, dataset_source):
        """
        Map back labels
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](box_features)

        del box_features
        if (self.unified_map_back or not self.training) and dataset_source != -1:            
            if self.training:
                pred_class_logits = pred_class_logits[:, self.dataset_inds[dataset_source]]
                for i in range(len(proposals)):
                    fg_inds = proposals[i].gt_classes != self.num_classes
                    proposals[i].gt_classes[fg_inds] = \
                        self.back_map[dataset_source][proposals[i].gt_classes[fg_inds]]
                    bg_inds = proposals[i].gt_classes == self.num_classes
                    proposals[i].gt_classes[bg_inds] = pred_class_logits.shape[1] - 1
            else:
                pred_class_logits = pred_class_logits[:, self.dataset_inds[dataset_source]]
        return pred_class_logits, pred_proposal_deltas
