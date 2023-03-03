# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
from collections import OrderedDict
from torch.functional import F


class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(Decoder, self).__init__()
        self.backbone_feature_reduction = nn.ModuleList()
        self.top_down_feature_reduction = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list[::-1]):
            self.backbone_feature_reduction.append(
                self._conv1x1_relu(in_channels, out_channels)
            )
            if i < len(in_channels_list) - 2:
                self.top_down_feature_reduction.append(
                    self._conv1x1_relu(out_channels, out_channels)
                )

    def _conv1x1_relu(self, in_channels, out_channels):
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=1,
                bias=False
            )),
            ("relu", nn.ReLU())
        ]))

    def forward(self, x):
        x = x[::-1]  # to lowest resolution first
        top_down_feature = None
        for i, feature in enumerate(x):
            feature = self.backbone_feature_reduction[i](feature)
            if i == 0:
                top_down_feature = feature
            else:
                upsampled_feature = F.interpolate(
                    top_down_feature,
                    size=feature.size()[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                if i < len(x) - 1:
                    top_down_feature = self.top_down_feature_reduction[i - 1](
                        feature + upsampled_feature
                    )
                else:
                    top_down_feature = feature + upsampled_feature
        return top_down_feature
