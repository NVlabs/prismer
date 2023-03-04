# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE
# Modified from: https://github.com/openai/CLIP/blob/main/clip/model.py

from collections import OrderedDict
from einops import rearrange
from clip.clip import _download

import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model.modules.utils import QuickGELU, LayerNorm, Adaptor, interpolate_pos_embed
from model.modules.resampler import PerceiverResampler
from huggingface_hub import hf_hub_download
from functools import partial


hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version='2.0.2')


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "ViT-H/14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ])
        )

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor, mode='attention'):
        if mode == 'attention':
            return x + self.attention(self.ln_1(x))
        elif mode == 'mlp':
            return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[nn.ModuleList([
            ResidualAttentionBlock(width, heads), 
            Adaptor(width),
        ]) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for resblock, adaptor in self.resblocks:
            x = resblock(x, mode='attention')
            x = adaptor(x)
            x = resblock(x, mode='mlp')
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, experts: dict):
        super().__init__()
        self.experts = experts

        self.conv1 = nn.ModuleDict()
        for e in experts:
            if e == 'rgb':
                self.conv1[e] = nn.Conv2d(in_channels=experts[e], out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
            elif e in ['seg', 'obj_detection', 'ocr_detection']:
                self.conv1[e] = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=4 / patch_size),
                    nn.Conv2d(in_channels=64, out_channels=width // 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width // 8),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 8, out_channels=width // 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width // 4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 4, out_channels=width // 2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(width // 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 2, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False),
                )
            else:
                self.conv1[e] = nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=16 / patch_size),
                    nn.Conv2d(in_channels=experts[e], out_channels=width // 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width // 8),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 8, out_channels=width // 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width // 4),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 4, out_channels=width // 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width // 2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width // 2, out_channels=width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False),
                )

        scale = width ** -0.5
        self.patch_size = patch_size
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, width))
        if 'obj_detection' in self.experts:
            self.instance_embedding = nn.Parameter(scale * torch.randn(128, width))
        self.transformer = Transformer(width, layers, heads)
        if len(self.experts) > 1:
            self.resampler = PerceiverResampler(width=width, layers=4, heads=8, num_latents=64)
        self.ln_pre = LayerNorm(width)
        self.ln_post = LayerNorm(width)

    def forward(self, x: dict):
        experts_inputs = []
        for exp in x:
            domain = 'seg' if 'seg' in exp else exp
            x_ = x[exp] if exp != 'obj_detection' else x[exp]['label']
            x_ = self.conv1[domain](x_)

            # add instance embedding (object detection only)
            if exp == 'obj_detection':
                instance_map = F.interpolate(x[exp]['instance'].to(x_.dtype), size=x_.shape[2:], mode='nearest')
                instance_map = rearrange(instance_map, 'b 1 h w -> b h w')
                label_map = rearrange(x_, 'b d h w -> d b h w')
                for l in x[exp]['instance'].unique():
                    l_ = random.randint(0, 127)
                    label_map[:, instance_map == l] += self.instance_embedding[l_].unsqueeze(-1)
                x_ = rearrange(label_map, 'd b h w -> b d h w')

            x_ = rearrange(x_, 'b d h w -> b (h w) d')

            # add position embedding (shared across all modalities)
            if domain == 'rgb':
                x_ = x_ + self.positional_embedding.to(x_.dtype)
                rgb_inputs = x_
            else:
                exp_positional_embedding = interpolate_pos_embed(self.positional_embedding.to(x_.dtype), x_.shape[1])
                x_ = x_ + exp_positional_embedding
                experts_inputs.append(x_)

        if len(experts_inputs) > 0:
            experts_inputs = rearrange(torch.cat(experts_inputs, dim=1), 'b l d -> l b d')
            experts_inputs = self.resampler(experts_inputs)
            rgb_inputs = rearrange(rgb_inputs, 'b l d -> l b d')
            x = torch.cat([rgb_inputs, experts_inputs], dim=0)
        else:
            x = rearrange(rgb_inputs, 'b l d -> l b d')

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)
        return x  # latents, batch, output_dim


def load_encoder(name: str, experts: dict, image_resolution: int):
    # load pre-trained model file
    if name in _MODELS:
        if name != 'ViT-H/14':
            model_path = _download(_MODELS[name], os.path.expanduser("cache/clip"))
            model = torch.jit.load(model_path, map_location="cpu")
            state_dict = model.state_dict()
        else:
            model_path = hf_hub_download(_MODELS[name], 'open_clip_pytorch_model.bin', revision=None, cache_dir="cache/clip")
            state_dict = torch.load(model_path, map_location="cpu")
    else:
        raise RuntimeError(f"Model {name} not found")

    # modify keys (we only need Vision Transformer)
    for key in list(state_dict.keys()):
        if not key.startswith('visual'):
            del state_dict[key]

    for key in list(state_dict.keys()):
        new_key = key.replace('visual.', '')
        if 'proj' in new_key and 'transformer' not in new_key:
            del state_dict[key]
        elif 'conv1' in new_key:
            new_key_ = new_key.replace('conv1', 'conv1.rgb')
            state_dict[new_key_] = state_dict.pop(key)
        elif 'positional_embedding' in new_key:
            state_dict[new_key] = state_dict.pop(key)[1:]
        elif 'transformer.resblocks' in new_key:
            new_key_ = re.sub(".mlp", ".0.mlp", new_key)
            new_key_ = re.sub(".attn", ".0.attn", new_key_)
            new_key_ = re.sub(".ln", ".0.ln", new_key_)
            state_dict[new_key_] = state_dict.pop(key)
        else:
            state_dict[new_key] = state_dict.pop(key)

    # load pre-trained weights
    vision_width = state_dict["conv1.rgb.weight"].shape[0]
    vision_patch_size = state_dict["conv1.rgb.weight"].shape[-1]
    vision_layers = len([k for k in state_dict.keys() if k.endswith(".attn.in_proj_weight")])
    vision_heads = vision_width // 64

    ViT = VisionTransformer(input_resolution=image_resolution,
                            patch_size=vision_patch_size,
                            width=vision_width,
                            layers=vision_layers,
                            heads=vision_heads,
                            experts=experts)

    state_dict['positional_embedding'] = interpolate_pos_embed(state_dict['positional_embedding'], len(ViT.positional_embedding))
    ViT.load_state_dict(state_dict, strict=False)
    return ViT


# Quick Check:
# model = load_encoder("ViT-B/16", experts={'rgb': 3, 'depth': 1, 'seg': 64}, image_resolution=224)
# rgb, depth, seg = torch.rand(4, 3, 224, 224), torch.rand(4, 1, 224, 224), torch.rand(4, 64, 224, 224)
# feat = model({'rgb': rgb, 'depth': depth, 'seg': seg})  # 260 [196 + 64], 4, 768
