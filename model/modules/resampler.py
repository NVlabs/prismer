# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import torch.nn as nn

from collections import OrderedDict
from einops import repeat
from model.modules.utils import SquaredReLU, LayerNorm


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)

        self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("sq_relu", SquaredReLU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.ln_ff = LayerNorm(d_model)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        return self.attn(q, kv, kv, need_weights=False)[0]

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        latents = latents + self.attention(q=self.ln_1(latents), kv=torch.cat([self.ln_1(latents), self.ln_2(x)], dim=0))
        latents = latents + self.mlp(self.ln_ff(latents))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_latents: int):
        super().__init__()
        scale = width ** -0.5
        self.latents = nn.Parameter(scale * torch.randn(num_latents, width))
        self.perceiver_blocks = nn.Sequential(*[PerceiverAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x_f: torch.Tensor):
        x = repeat(self.latents, 'l d -> l b d', b=x_f.shape[1])

        for p_block in self.perceiver_blocks:
            x = p_block(x_f, x)

        return x  # num_latents, batch_size, output_dim


# Quick Check:
# resampler = PerceiverResampler(width=768, layers=6, heads=8, num_latents=64)
# feat = torch.rand(4, 256, 768)
# expert_feat = resampler(feat)  # 64, 256, 768
