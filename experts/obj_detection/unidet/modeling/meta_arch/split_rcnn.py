import logging
import numpy as np
import torch
import json
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads


@META_ARCH_REGISTRY.register()
class SplitClassifierRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.datasets = cfg.MULTI_DATASET.DATASETS
        self.num_datasets = len(self.datasets)
        self.dataset_name_to_id = {k: i for i, k in enumerate(self.datasets)}
        self.eval_dataset = -1

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for i in range(len(gt_instances)):
            dataset_source = batched_inputs[i]['dataset_source']
            gt_instances[i]._dataset_source = dataset_source

        features = self.backbone(images.tensor) # #lvl
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, 
        do_postprocess=True):
        assert not self.training
        assert detected_instances is None
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(
            images, features, proposals, None, eval_dataset=self.eval_dataset)
        
        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def set_eval_dataset(self, dataset_name):
        meta_datase_name = dataset_name[:dataset_name.find('_')]
        self.eval_dataset = \
            self.dataset_name_to_id[meta_datase_name]
