# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# customised LayerNorm
class LayerNorm(nn.LayerNorm):
    # We always use float32 for the LayerNorm for stable training
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight.to(torch.float32), self.bias.to(torch.float32), self.eps)
        return ret.type(orig_type)


# activations
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


# interpolate position embedding
def interpolate_pos_embed(orig_pos_embed, target_len):
    orig_size = int((orig_pos_embed.shape[0]) ** 0.5)
    new_size = int(target_len ** 0.5)

    if orig_size != new_size:
        orig_pos_embed = orig_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        orig_pos_embed = F.interpolate(orig_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
        orig_pos_embed = orig_pos_embed.permute(0, 2, 3, 1).flatten(0, 2)
        return orig_pos_embed
    else:
        return orig_pos_embed


# Adaptor design
class Adaptor(nn.Module):
    def __init__(self, embed_dim: int, norm_late=False):
        super().__init__()
        self.norm_late = norm_late
        self.adaptor = nn.Sequential(OrderedDict([
                ("down_proj", nn.Linear(embed_dim, embed_dim // 1)),
                ("sq_relu", SquaredReLU()),
                ("up_proj", nn.Linear(embed_dim // 1, embed_dim))
            ])
        )
        self.adaptor_ln = LayerNorm(embed_dim)

    def forward(self, hidden_states: torch.Tensor):
        if self.norm_late:
            hidden_states = self.adaptor_ln(self.adaptor(hidden_states) + hidden_states)
        else:
            hidden_states = self.adaptor(self.adaptor_ln(hidden_states)) + hidden_states
        return hidden_states
