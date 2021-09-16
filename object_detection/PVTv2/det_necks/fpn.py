# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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


"""FPN Lyaer for object detection"""
import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import XavierUniform
import paddle.nn.functional as F


class ConvNorm(nn.Layer):
    """ Conv + BatchNorm (optional) layers
    Args:
        in_channels: int, num of input channels 
        out_channels: int, num of output channels
        kernel_size: int, conv kernel size
        stride: int, stride in conv layer, default: 1
        padding: int, padding in conv layer, default: 0 
        dilation: int, dilation in conv layer, default: 1 
        groups: int, groups in conv layer, default: 1 
        padding_mode: str, padding mode, default: 'zeros' 
        weight_attr: ParamAttr, paddle param setting for weight, default: None 
        bias_attr: ParamAttr, paddle param setting for bias, default: None
        norm: string, type of norm layer, default: bn
    """
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
                 norm="bn",
                 use_bias=False):
        super(ConvNorm, self).__init__()

        if norm is None:
            use_bias = None

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
    """Feature Pyramid Network (FPN) Layer
    Args:
        in_channels: list of int, num of input channels for each output layer
        out_channels: list of int, num of output channels for each output layer
        stride: list, spatial strides between each feature layer to the original image size
        fuse_type: str, how to fuse current and prev feature in FPN, avg or sum, default: sum
        use_c5: bool, if True, use C5 as the input of extra stage, default: True
        top_block: nn.Layer, if use a downsample after output (see LastLevelMaxPool), default: None
        norm: str, type of norm layer, default: None
    """
    def __init__(self,
                 in_channels,
                 out_channel,
                 strides,
                 fuse_type="sum",
                 use_c5=True,
                 top_block=None,
                 norm=None,
                 use_bias=False):
        super(FPN, self).__init__()
        assert len(strides) == len(in_channels)

        self.fuse_type = fuse_type
        self.top_block = top_block
        self.use_c5 = use_c5

        lateral_convs = []
        output_convs = []

        name_idx = [int(math.log2(s)) for s in strides]

        for idx, in_channel in enumerate(in_channels):
            # 1x1 conv 
            lateral_conv = ConvNorm(
                in_channels=in_channel, 
                out_channels=out_channel, 
                kernel_size=1,
                weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=in_channel)),
                norm=norm,
                use_bias=use_bias)
            # 3x3 conv after upsampling
            output_conv = ConvNorm(
                in_channels=out_channel, 
                out_channels=out_channel, 
                kernel_size=3,
                padding=1,
                weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_out=9*out_channel)),
                norm=norm,
                use_bias=use_bias)

            self.add_sublayer("fpn_lateral{}".format(name_idx[idx]), lateral_conv)
            self.add_sublayer("fpn_output{}".format(name_idx[idx]), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        self.lateral_convs = lateral_convs[::-1] # Now from small feature map to large feature map
        self.output_convs = output_convs[::-1]

    def forward(self, feats):
        res = []
        lateral_out = self.lateral_convs[0](feats[-1]) # feats is from large to small feature map
        res.append(self.output_convs[0](lateral_out))

        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)):
            if idx > 0:  # not include lateral_convs[0]
                top2down_feat = F.interpolate(lateral_out, scale_factor=2.0, mode="nearest")
                prev_out = lateral_conv(feats[-1-idx])
                #top2down_feat = F.interpolate(lateral_out, size=prev_out.shape[-2::], mode="nearest")
                lateral_out = prev_out + top2down_feat # fuse == 'sum'
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
