from detectron2.config import CfgNode as CN

def add_unidet_config(cfg):
    _C = cfg

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False # Equalization loss described in https://arxiv.org/abs/2003.05176
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/oid/annotations/openimages_challenge_2019_train_v2_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False # Federated loss described in https://www.lvisdataset.org/assets/challenge_reports/2020/CenterNet2.pdf, not used in this project
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_PATH = \
        'datasets/oid/annotations/challenge-2019-label500-hierarchy-list.json' # Hierarchical-loss for OpenImages
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_IGNORE = False # Ignore child classes when the annotation is an internal class
    _C.MODEL.ROI_BOX_HEAD.HIERARCHY_POS_PARENTS = False # Set parent classes in the hierarchical tree as positive 
    _C.MODEL.ROI_BOX_HEAD.UNIFIED_MAP_BACK = True # Ignore out-of-dataset classes for retraining unified model
    _C.MODEL.ROI_BOX_HEAD.FIX_NORM_REG = False # not used
    
    # ResNeSt
    _C.MODEL.RESNETS.DEEP_STEM = False
    _C.MODEL.RESNETS.AVD = False
    _C.MODEL.RESNETS.AVG_DOWN = False
    _C.MODEL.RESNETS.RADIX = 1
    _C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

    _C.MULTI_DATASET = CN()
    _C.MULTI_DATASET.ENABLED = False
    _C.MULTI_DATASET.DATASETS = ['objects365', 'coco', 'oid']
    _C.MULTI_DATASET.NUM_CLASSES = [365, 80, 500]
    _C.MULTI_DATASET.DATA_RATIO = [1, 1, 1]
    _C.MULTI_DATASET.UNIFIED_LABEL_FILE = ''
    _C.MULTI_DATASET.UNIFY_LABEL_TEST = False # convert the partitioned model to a unified model at test time
    _C.MULTI_DATASET.UNIFIED_EVAL = False
    _C.MULTI_DATASET.SAMPLE_EPOCH_SIZE = 1600
    _C.MULTI_DATASET.USE_CAS = [False, False, False] # class-aware sampling
    _C.MULTI_DATASET.CAS_LAMBDA = 1. # Class aware sampling weight from https://arxiv.org/abs/2005.08455, not used in this project
    _C.MULTI_DATASET.UNIFIED_NOVEL_CLASSES_EVAL = False # zero-shot cross dataset evaluation
    _C.MULTI_DATASET.MATCH_NOVEL_CLASSES_FILE = '' 
    

    _C.SOLVER.RESET_ITER = False # used when loading a checkpoint for finetuning
    _C.CPU_POST_PROCESS = False
    _C.TEST.AUG.NMS_TH = 0.7
    _C.DEBUG = False
    _C.VIS_THRESH = 0.3
    _C.DUMP_CLS_SCORE = False # dump prediction logits to disk. Used for the distortion metric for learning a label space
    _C.DUMP_BBOX = False
    _C.DUMP_NUM_IMG = 2000
    _C.DUMP_NUM_PER_IMG = 50
