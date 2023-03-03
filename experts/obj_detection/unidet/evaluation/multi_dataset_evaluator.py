# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, defaultdict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import glob
from PIL import Image

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

from .oideval import OIDEvaluator, _evaluate_predictions_on_oid

def get_unified_evaluator(
    evaluator_type, 
    dataset_name, cfg, distributed, output_dir):
    unified_label_file = cfg.MULTI_DATASET.UNIFIED_LABEL_FILE
    if evaluator_type == 'coco':
        evaluator = UnifiedCOCOEvaluator(
            unified_label_file,
            dataset_name, cfg, distributed, output_dir)
    elif evaluator_type == 'oid':
        evaluator = UnifiedOIDEvaluator(
            unified_label_file,
            dataset_name, cfg, distributed, output_dir)
    elif evaluator_type == 'cityscapes_instance':
        evaluator = UnifiedCityscapesEvaluator(
            unified_label_file,
            dataset_name, cfg, distributed, output_dir)
    else:
        assert 0, evaluator_type
    return evaluator


def map_back_unified_id(results, map_back, reverse_id_mapping=None):
    ret = []
    for result in results:
        if result['category_id'] in map_back:
            result['category_id'] = map_back[result['category_id']]
            if reverse_id_mapping is not None:
                result['category_id'] = reverse_id_mapping[result['category_id']]
            ret.append(result)
    return ret


def map_back_unified_id_novel_classes(results, map_back, reverse_id_mapping=None):
    ret = []
    for result in results:
        if result['category_id'] in map_back:
            original_id_list = map_back[result['category_id']]
            for original_id in original_id_list:
                result_copy = copy.deepcopy(result)
                result_copy['category_id'] = original_id
                if reverse_id_mapping is not None:
                    result_copy['category_id'] = \
                        reverse_id_mapping[result_copy['category_id']]
                ret.append(result_copy)
    return ret

class UnifiedCOCOEvaluator(COCOEvaluator):
    def __init__(
        self, unified_label_file, dataset_name, cfg, 
        distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir)
        meta_dataset_name = dataset_name[:dataset_name.find('_')]
        print('meta_dataset_name', meta_dataset_name)
        self.meta_dataset_name = meta_dataset_name
        self._logger.info("saving outputs to {}".format(self._output_dir))
        self.unified_novel_classes_eval = cfg.MULTI_DATASET.UNIFIED_NOVEL_CLASSES_EVAL
        if self.unified_novel_classes_eval:
            match_novel_classes_file = cfg.MULTI_DATASET.MATCH_NOVEL_CLASSES_FILE

            print('Loading map back from', match_novel_classes_file)
            novel_classes_map = json.load(
                open(match_novel_classes_file, 'r'))[meta_dataset_name]
            self.map_back = {}
            for c, match in enumerate(novel_classes_map):
                for m in match:
                    # one ground truth label may be maped back to multiple original labels
                    if m in self.map_back:
                        self.map_back[m].append(c)
                    else:
                        self.map_back[m] = [c]
        else:
            unified_label_data = json.load(open(unified_label_file, 'r'))
            label_map = unified_label_data['label_map']
            label_map = label_map[meta_dataset_name]
            self.map_back = {int(v): i for i, v in enumerate(label_map)}

    def _eval_predictions(self, tasks, predictions):
        self._logger.info("Preparing results for COCO format ...")
        _unified_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        file_path = os.path.join(
            self._output_dir, "unified_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(_unified_results))
            f.flush()

        assert hasattr(self._metadata, "thing_dataset_id_to_contiguous_id")
        reverse_id_mapping = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }

        if self.unified_novel_classes_eval:
            self._coco_results = map_back_unified_id_novel_classes(
                _unified_results, self.map_back, 
                reverse_id_mapping=reverse_id_mapping)
        else:
            self._coco_results = map_back_unified_id(
                _unified_results, self.map_back, 
                reverse_id_mapping=reverse_id_mapping)

        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(self._coco_results))
            f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

class UnifiedCityscapesEvaluator(COCOEvaluator):
    def __init__(
        self, unified_label_file, dataset_name, cfg, 
        distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir)
        meta_dataset_name = dataset_name[:dataset_name.find('_')]
        print('meta_dataset_name', meta_dataset_name)

        self.unified_novel_classes_eval = cfg.MULTI_DATASET.UNIFIED_NOVEL_CLASSES_EVAL
        if self.unified_novel_classes_eval:
            match_novel_classes_file = cfg.MULTI_DATASET.MATCH_NOVEL_CLASSES_FILE
            print('Loading map back from', match_novel_classes_file)
            novel_classes_map = json.load(
                open(match_novel_classes_file, 'r'))[meta_dataset_name]
            self.map_back = {}
            for c, match in enumerate(novel_classes_map):
                for m in match:
                    self.map_back[m] = c
        else:
            unified_label_data = json.load(open(unified_label_file, 'r'))
            label_map = unified_label_data['label_map']
            label_map = label_map[meta_dataset_name]
            self.map_back = {int(v): i for i, v in enumerate(label_map)}

        self._logger.info("saving outputs to {}".format(self._output_dir))
        self._temp_dir = self._output_dir + '/cityscapes_style_eval_tmp/'
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )
        PathManager.mkdirs(self._temp_dir)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {
                "image_id": input["image_id"], 
                "file_name": input['file_name']
            }

            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            for x in prediction["instances"]:
                x['file_name'] = input['file_name']
            # if len(prediction['instances']) == 0:
            #     self._logger.info("No prediction for {}".format(x['file_name']))
            #     prediction['instances'] = [
            #         {'file_name': input['file_name'], 
            #         ''}]
            self._predictions.append(prediction)

    def _eval_predictions(self, tasks, predictions):
        self._logger.info("Preparing results for COCO format ...")
        _unified_results = list(itertools.chain(
            *[x["instances"] for x in predictions]))
        all_file_names = [x['file_name'] for x in predictions]
        file_path = os.path.join(
            self._output_dir, "unified_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(_unified_results))
            f.flush()

        mapped = False
        thing_classes = None
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            self._logger.info('Evaluating COCO-stype cityscapes! '+ \
                'Using buildin meta to mapback IDs.')
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            mapped = True
            thing_classes = {
                k: self._metadata.thing_classes[v] \
                    for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()}
        else:
            self._logger.info('Evaluating cityscapes! '+ \
                'Using eval script to map back IDs.')
            reverse_id_mapping = None
            thing_classes = self._metadata.thing_classes

        if self.unified_novel_classes_eval:
            coco_results = map_back_unified_id_novel_classes(
                _unified_results, self.map_back, 
                reverse_id_mapping=reverse_id_mapping)
        else:
            coco_results = map_back_unified_id(
                _unified_results, self.map_back, 
                reverse_id_mapping=reverse_id_mapping)

        self.write_as_cityscapes(
            coco_results, all_file_names, 
            temp_dir=self._temp_dir, mapped=mapped, 
            thing_classes=thing_classes)

        os.environ["CITYSCAPES_DATASET"] = os.path.abspath(
            os.path.join(self._metadata.gt_dir, "..", "..")
        )
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))
        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        return ret

    @staticmethod
    def write_as_cityscapes(coco_results, all_file_names, 
        temp_dir, mapped=False, thing_classes=None, 
        ext='_pred.txt', subfolder=''):
        from cityscapesscripts.helpers.labels import name2label
        results_per_image = {x: [] for x in all_file_names}
        for x in coco_results:
            results_per_image[x['file_name']].append(x)
        if subfolder != '':
            PathManager.mkdirs(temp_dir + '/' + subfolder)
        N = len(results_per_image)
        for i, (file_name, coco_list) in enumerate(results_per_image.items()):
            if i % (N // 10) == 0:
                print('{}%'.format(i // (N // 10) * 10), end=',', flush=True)
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(temp_dir, basename + ext)

            num_instances = len(coco_list)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    if not mapped:
                        pred_class = coco_list[i]['category_id']
                        classes = thing_classes[pred_class]
                        class_id = name2label[classes].id
                    else:
                        class_id = coco_list[i]['category_id']
                        classes = thing_classes[class_id]
                    score = coco_list[i]['score']
                    mask = mask_util.decode(coco_list[i]['segmentation'])[:, :].astype("uint8")
                    # mask = output.pred_masks[i].numpy().astype("uint8")
                    if subfolder != '':
                        png_filename = os.path.join(
                            temp_dir, subfolder, basename + "_{}_{}.png".format(
                                i, classes.replace(' ', '_'))
                        )
                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write("{} {} {}\n".format(
                            subfolder + '/' + os.path.basename(png_filename), class_id, score))

                    else:
                        png_filename = os.path.join(
                            temp_dir, basename + "_{}_{}.png".format(i, classes.replace(' ', '_'))
                        )

                        Image.fromarray(mask * 255).save(png_filename)
                        fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))


class UnifiedOIDEvaluator(OIDEvaluator):
    def __init__(
        self, unified_label_file, dataset_name, cfg, 
        distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir)
        meta_dataset_name = dataset_name[:dataset_name.find('_')]
        print('meta_dataset_name', meta_dataset_name)
        unified_label_data = json.load(open(unified_label_file, 'r'))
        label_map = unified_label_data['label_map']
        label_map = label_map[meta_dataset_name]
        self.map_back = {int(v): i for i, v in enumerate(label_map)}
        self._logger.info("saving outputs to {}".format(self._output_dir))

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return

        if len(self._predictions) == 0:
            self._logger.warning("[LVISEvaluator] Did not receive valid predictions.")
            return {}

        self._logger.info("Preparing results in the OID format ...")
        _unified_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        file_path = os.path.join(
            self._output_dir, "unified_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(_unified_results))
            f.flush()

        self._oid_results = map_back_unified_id(
            _unified_results, self.map_back)

        # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
        for result in self._oid_results:
            result["category_id"] += 1

        PathManager.mkdirs(self._output_dir)
        file_path = os.path.join(
            self._output_dir, "oid_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(self._oid_results))
            f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        self._results = OrderedDict()
        res = _evaluate_predictions_on_oid(
            self._oid_api,
            file_path,
            eval_seg=self._mask_on
        )
        self._results['bbox'] = res

        return copy.deepcopy(self._results)


