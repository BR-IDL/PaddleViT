#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle
import paddle.nn as nn
from paddle.nn.initializer import XavierUniform
import paddle.nn.functional as F

class ConvNorm(nn.Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 padding_mode='zeros', 
                 weight_attr=None, 
                 bias_attr=None,
                 norm=""):
        super(ConvNorm, self).__init__()

        use_bias = None if norm == "" else False

        self.conv = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            padding_mode=padding_mode, 
            weight_attr=weight_attr, 
            bias_attr=use_bias
        )

        if norm == "bn":
            self.norm = nn.BatchNorm2D(out_channels)
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.norm(out)
        
        return out


class FPN(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channel,
        strides,
        fuse_type="sum",
        use_c5=True,
        top_block=None,
        norm=""
    ):
        super(FPN, self).__init__()

        assert len(strides) == len(in_channels)

        self.fuse_type = fuse_type
        self.top_block = top_block
        self.use_c5 = use_c5

        lateral_convs = []
        output_convs = []

        name_idx = [int(math.log2(s)) for s in strides]

        for idx, in_channel in enumerate(in_channels):
            lateral_conv = ConvNorm(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=1,
                weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=in_channel)),
                norm=norm
            )
            output_conv = ConvNorm(
                in_channels=out_channel, 
                out_channels=out_channel, 
                kernel_size=3,
                padding=1,
                weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=9*out_channel)),
                norm=norm
            )
            self.add_sublayer("fpn_lateral{}".format(name_idx[idx]), lateral_conv)
            self.add_sublayer("fpn_output{}".format(name_idx[idx]), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, feats):
        res = []
        lateral_out = self.lateral_convs[0](feats[-1])
        res.append(self.output_convs[0](lateral_out))

        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            if idx > 0:  # not include lateral_convs[0]
                top2down_feat = F.interpolate(lateral_out, scale_factor=2.0, mode="nearest")
                prev_out = lateral_conv(feats[-1-idx])
                lateral_out = prev_out + top2down_feat
                if self.fuse_type == "avg":
                    lateral_out /= 2
                res.insert(0, output_conv(lateral_out))
        
        if self.top_block is not None:
            if self.use_c5:
                top_block_out = self.top_block(feats[-1])
            else:
                top_block_out = self.top_block(res[-1])
        
            res.extend(top_block_out)

        return res


class LastLevelMaxPool(nn.Layer):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2)]


class TopFeatP6P7(nn.Layer):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """
    def __init__(self, in_channel, out_channel):

        self.p6 = nn.Conv2D(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=3, 
            stride=2, 
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=9*in_channel))
        )
        self.p7 = nn.Conv2D(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=3, 
            stride=2, 
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=9*out_channel))
        )
    
    def forward(self, feat):
        p6 = self.p6(feat)
        p7 = self.p7(F.relu(p6))

        return [p6, p7]

