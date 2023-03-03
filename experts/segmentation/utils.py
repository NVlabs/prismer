from experts.segmentation.mask2former import add_maskformer2_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.freeze()
    return cfg
