#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

"""
Implement The all MLP Head of Segformer
Segformer: https://arxiv.org/abs/2105.15203 

Adapted from repository below:
    https://github.com/open-mmlab/mmsegmentation
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ConvModule(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvModule, self).__init__()

        norm_bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        norm_weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=1.0))

        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              bias_attr=False)
        self.bn = nn.BatchNorm(out_channels,
                               param_attr=norm_weight_attr,
                               bias_attr=norm_bias_attr)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class SegformerHead(nn.Layer):
    def __init__(self, in_channels, channels, num_classes, align_corners):
        super().__init__()

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        self.conv_seg = nn.Conv2D(self.channels, self.num_classes, 1, stride=1)
        self.convs = nn.LayerList()
        num_inputs = len(self.in_channels)

        for i in range(num_inputs):
            self.convs.append(
                ConvModule(self.in_channels[i], self.channels, 1, 1))

        self.fusion_conv = ConvModule(self.channels * num_inputs,
                                      self.channels, 1, 1)

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(
            value=1.0))

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.convs[idx](x)
            outs.append(
                F.interpolate(x,
                              inputs[0].shape[2:],
                              mode='bilinear',
                              align_corners=self.align_corners))
        out = self.fusion_conv(paddle.concat(outs, axis=1))

        out = self.conv_seg(out)
        up4x_resolution = [4 * item for item in inputs[0].shape[2:]]
        out = F.interpolate(out,
                            up4x_resolution,
                            mode='bilinear',
                            align_corners=self.align_corners)
        return [out]
