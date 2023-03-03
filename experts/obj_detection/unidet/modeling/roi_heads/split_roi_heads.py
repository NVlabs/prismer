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
class MultiDatasetCascadeROIHeads(CustomCascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        self.dataset_names = cfg.MULTI_DATASET.DATASETS
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                MultiDatasetFastRCNNOutputLayers(
                    cfg,
                    cfg.MULTI_DATASET.NUM_CLASSES,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
        ret['box_predictors'] = box_predictors

        self.unify_label_test = cfg.MULTI_DATASET.UNIFY_LABEL_TEST
        if self.unify_label_test:
            unified_label_data = json.load(
                open(cfg.MULTI_DATASET.UNIFIED_LABEL_FILE, 'r'))
            label_map = unified_label_data['label_map']
            self.label_map = {
                d: torch.tensor(x).long().to(torch.device(cfg.MODEL.DEVICE)) \
                for d, x in label_map.items()}
            self.unified_num_class = len(set().union(
                *[label_map[d] for d in label_map]))
            # add background class
            self.label_map = {d: torch.cat([
                self.label_map[d], 
                self.label_map[d].new_tensor([self.unified_num_class])]) for d in label_map}
            self.class_count = torch.zeros(self.unified_num_class + 1).float().to(
                    torch.device(cfg.MODEL.DEVICE))
            for d in self.label_map:
                self.class_count[self.label_map[d]] = \
                    self.class_count[self.label_map[d]] + 1

        self.dump_cls_score = cfg.DUMP_CLS_SCORE
        if self.dump_cls_score:
            self.dump_num_img = cfg.DUMP_NUM_IMG
            self.dump_num_per_img = cfg.DUMP_NUM_PER_IMG
            self.class_scores = []
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
                    stage_losses = predictor.losses(
                        predictions, proposals, dataset_source)
                losses.update({"{}_{}_stage{}".format(
                    self.dataset_names[dataset_source], 
                    k, stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
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
        support dataset_source
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)

        if self.unify_label_test and not self.training:
            pred_class_logits_all, pred_proposal_deltas = self.box_predictor[stage](
                box_features, -1)
            unified_score = pred_proposal_deltas.new_zeros(
                    (pred_class_logits_all[0].shape[0], self.unified_num_class + 1))
            for i, d in enumerate(self.dataset_names):
                pred_class_score = pred_class_logits_all[i]
                unified_score[:, self.label_map[d]] = \
                    unified_score[:, self.label_map[d]] + pred_class_score
            unified_score = unified_score / self.class_count
            if dataset_source in self.dataset_names:
                # on training datasets
                pred_class_logits = \
                    unified_score[:, self.label_map[self.dataset_names[dataset_source]]]
            else:
                pred_class_logits = unified_score
            # B x (#U + 1)
        else:
            pred_class_logits, pred_proposal_deltas = self.box_predictor[stage](
                box_features, dataset_source if type(dataset_source) != type('') else -1)
            if not self.training and (dataset_source == -1 or type(dataset_source) == type('')):
                fg = torch.cat(
                    [x[:, :-1] for x in pred_class_logits], dim=1)
                bg = torch.cat(
                    [x[:, -1:] for x in pred_class_logits], dim=1).mean(dim=1)
                pred_class_logits = torch.cat([fg, bg[:, None]], dim=1)
                # B x (sum C + 1)

        if self.dump_cls_score:
            if not self.unify_label_test:
                pred_class_logits_all, _ = self.box_predictor[stage](
                    box_features, -1)
            if len(self.class_scores) < self.dump_num_img and stage == 2:
                self.class_scores.append(
                    [x[:self.dump_num_per_img].detach().cpu().numpy() \
                        for x in pred_class_logits_all])

        return pred_class_logits, pred_proposal_deltas
