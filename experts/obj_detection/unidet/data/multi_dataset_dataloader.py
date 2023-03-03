import copy
import logging
import numpy as np
import operator
import torch.utils.data
import json
from detectron2.utils.comm import get_world_size

from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
from collections import defaultdict
from typing import Optional


def get_detection_dataset_dicts_with_source(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    Similar to detectron2.data.build.get_detection_dataset_dicts, but also returns the dataset
        source.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    
    for source_id, (dataset_name, dicts) in \
        enumerate(zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        for d in dicts:
            d['dataset_source'] = source_id

        if "annotations" in dicts[0]:
            try:
                class_names = MetadataCatalog.get(dataset_name).thing_classes
                check_metadata_consistency("thing_classes", dataset_name)
                print_instances_class_histogram(dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

    assert proposal_files is None

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    return dataset_dicts

def build_multi_dataset_train_loader(cfg, mapper=None):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts_with_source(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
    for d in dataset_dicts:
        sizes[d['dataset_source']] += 1
    # print('sizes', sizes)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == 'MultiDatasetSampler':
        sampler = MultiDatasetSampler(cfg, dataset_dicts, sizes)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    assert cfg.DATALOADER.ASPECT_RATIO_GROUPING

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict

    data_loader = MDAspectRatioGroupedDataset(
        data_loader, images_per_worker, num_datasets=len(sizes))

    return data_loader


class MultiDatasetSampler(Sampler):
    def __init__(self, cfg, dataset_dicts, sizes, seed: Optional[int] = None):
        """
        """
        self.sizes = sizes
        self.sample_epoch_size = cfg.MULTI_DATASET.SAMPLE_EPOCH_SIZE
        assert self.sample_epoch_size % cfg.SOLVER.IMS_PER_BATCH == 0
        print('self.epoch_size', self.sample_epoch_size)
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._batch_size = cfg.SOLVER.IMS_PER_BATCH
        self._ims_per_gpu = self._batch_size // self._world_size

        self.dataset_ids =  torch.tensor(
            [d['dataset_source'] for d in dataset_dicts], dtype=torch.long)
        st = 0

        dataset_ratio = cfg.MULTI_DATASET.DATA_RATIO
        assert len(dataset_ratio) == len(sizes), \
            'length of dataset ratio {} should be equal to number if dataset {}'.format(
                len(dataset_ratio), len(sizes)
            )
        dataset_weight = [torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio) \
            for i, (r, s) in enumerate(zip(dataset_ratio, sizes))]
        st = 0
        cas_factors = []
        for i, s in enumerate(sizes):
            if cfg.MULTI_DATASET.USE_CAS[i]:
                cas_factor = self._get_class_balance_factor_per_dataset(
                    dataset_dicts[st: st + s],
                    l=cfg.MULTI_DATASET.CAS_LAMBDA)
                cas_factor = cas_factor * (s / cas_factor.sum())
            else:
                cas_factor = torch.ones(s)
            cas_factors.append(cas_factor)
            st = st + s
        cas_factors = torch.cat(cas_factors)
        dataset_weight = torch.cat(dataset_weight)
        self.weights = dataset_weight * cas_factors


    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size)


    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            ids = torch.multinomial(
                self.weights, self.sample_epoch_size, generator=g, 
                replacement=True)
            # nums = [(self.dataset_ids[ids] == i).sum().int().item() \
            #     for i in range(len(self.sizes))]
            # print('_rank, len, nums, self.dataset_ids[ids[:10]], ', 
            #     self._rank, len(ids), nums, self.dataset_ids[ids[:10]], 
            #     flush=True)
            yield from ids


    def _get_class_balance_factor_per_dataset(self, dataset_dicts, l=1.):
        ret = []
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for i, dataset_dict in enumerate(dataset_dicts):
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            ret.append(sum(
                [1. / (category_freq[cat_id] ** l) for cat_id in cat_ids]))
        return torch.tensor(ret).float()

class MDAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    """
    """

    def __init__(self, dataset, batch_size, num_datasets):
        """
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2 * num_datasets)]


    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
