# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import torchvision.transforms as transforms


def load_expert_model(task=None):
    if task == 'depth':
        # DPT model is a standard pytorch model class
        from experts.depth.models import DPTDepthModel

        model = DPTDepthModel(path='experts/expert_weights/dpt_hybrid-midas-501f0c75.pt',
                              backbone="vitb_rn50_384",
                              non_negative=True,
                              enable_attention_hooks=False)
        transform = transforms.Compose([
            transforms.Resize([480, 480]),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]
        )

    elif task == 'seg_coco':
        # Mask2Former is wrapped in detection2,
        # the model takes input in the format of: {"image": image (BGR), "height": height, "width": width}
        import argparse
        from detectron2.engine.defaults import DefaultPredictor
        from experts.segmentation.utils import setup_cfg

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default="client")
        parser.add_argument("--port", default=2)
        args = parser.parse_args()

        args.config_file = 'experts/segmentation/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml'
        args.opts = ['MODEL.WEIGHTS', 'experts/expert_weights/model_final_f07440.pkl']
        cfg = setup_cfg(args)
        model = DefaultPredictor(cfg).model
        transform = transforms.Compose([
            transforms.Resize(size=479, max_size=480)
        ])

    elif task == 'seg_ade':
        # Mask2Former is wrapped in detection2,
        # the model takes input in the format of: {"image": image (BGR), "height": height, "width": width}
        import argparse
        from detectron2.engine.defaults import DefaultPredictor
        from experts.segmentation.utils import setup_cfg

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default="client")
        parser.add_argument("--port", default=2)
        args = parser.parse_args()

        args.config_file = 'experts/segmentation/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml'
        args.opts = ['MODEL.WEIGHTS', 'experts/expert_weights/model_final_e0c58e.pkl']
        cfg = setup_cfg(args)
        model = DefaultPredictor(cfg).model
        transform = transforms.Compose([
            transforms.Resize(size=479, max_size=480)
        ])

    elif task == 'obj_detection':
        # UniDet is wrapped in detection2,
        # the model takes input in the format of: {"image": image (BGR), "height": height, "width": width}
        import argparse
        from detectron2.engine.defaults import DefaultPredictor
        from experts.obj_detection.utils import setup_cfg
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default="client")
        parser.add_argument("--port", default=2)
        parser.add_argument("--confidence-threshold", type=float, default=0.5)
        args = parser.parse_args()

        args.config_file = 'experts/obj_detection/configs/Unified_learned_OCIM_RS200_6x+2x.yaml'
        args.opts = ['MODEL.WEIGHTS', 'experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth']

        cfg = setup_cfg(args)
        model = DefaultPredictor(cfg).model
        transform = transforms.Compose([
            transforms.Resize(size=479, max_size=480)
        ])

    elif task == 'ocr_detection':
        from experts.ocr_detection.charnet.modeling.model import CharNet
        model = CharNet()
        model.load_state_dict(torch.load('experts/expert_weights/icdar2015_hourglass88.pth'))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif task == 'normal':
        # NLL-AngMF model is a standard pytorch model class
        import argparse
        from experts.normal.models.NNET import NNET
        from experts.normal.utils import utils

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default="client")
        parser.add_argument("--port", default=2)
        parser.add_argument('--architecture', default='BN', type=str, help='{BN, GN}')
        parser.add_argument("--pretrained", default='scannet', type=str, help="{nyu, scannet}")
        parser.add_argument('--sampling_ratio', type=float, default=0.4)
        parser.add_argument('--importance_ratio', type=float, default=0.7)
        args = parser.parse_args()
        model = NNET(args)
        model = utils.load_checkpoint('experts/expert_weights/scannet.pt', model)

        transform = transforms.Compose([
            transforms.Resize([480, 480]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif task == 'edge':
        # NLL-AngMF model is a standard pytorch model class
        from experts.edge.model import DexiNed
        model = DexiNed()
        model.load_state_dict(torch.load('experts/expert_weights/10_model.pth', map_location='cpu'))
        transform = transforms.Compose([
            transforms.Resize([480, 480]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])
        ])
    else:
        print('Task not supported')
        model = None
        transform = None

    model.eval()
    return model, transform




