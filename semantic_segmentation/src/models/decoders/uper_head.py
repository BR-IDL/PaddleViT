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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .psp_head import PyramidPoolingModule 


class UperHead(nn.Layer):
    """UperHead

    UperHead is the decoder of UperNet, Ref https://arxiv.org/pdf/1807.10221.pdf
    Reference:                                                                                                                                                
        Tete Xiao, et al. *"Unified Perceptual Parsing for Scene Understanding"*
    """
   
    def __init__(self, pool_scales, in_channels, channels, align_corners=False, num_classes=60):
        super(UperHead, self).__init__()
        self.pool_scales = pool_scales
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)) 
        # PSP module
        self.psp_modules = PyramidPoolingModule(self.pool_scales, self.in_channels[-1], self.channels, self.align_corners)
        self.bottleneck = nn.Sequential(
            nn.Conv2D(self.in_channels[-1] + len(self.pool_scales)*self.channels, \
                self.channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
            nn.ReLU())
        # FPN module
        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()
        for in_channel in self.in_channels[:-1]: # skip the top layer
           l_conv = nn.Sequential(
                nn.Conv2D(in_channel, self.channels, kernel_size=1, stride=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())
           fpn_conv = nn.Sequential(
                nn.Conv2D(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())
           self.lateral_convs.append(l_conv)
           self.fpn_convs.append(fpn_conv)
        # FPN bottleneck
        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2D(len(self.in_channels)*self.channels, self.channels, kernel_size=3,\
                    stride=1, padding=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())
        self.conv_seg = nn.Conv2D(self.channels, self.num_classes, 1, stride=1)


    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def psp_forward(self, x):
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = paddle.concat(psp_outs, axis=1)  
        out = self.bottleneck(psp_outs) 
        return out

    def forward(self, inputs):
        up4x_resolution = [ 4*item for item in inputs[0].shape[2:]]
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs[-1]))
        # build top-down fusion
        used_feat_levels = len(laterals)
        for idx in range(used_feat_levels-1, 0, -1):
           prev_size = laterals[idx-1].shape[2:]
           laterals[idx-1] += F.interpolate(laterals[idx], prev_size, mode='bilinear', align_corners=self.align_corners) 
        # build fpn-output
        fpn_outs = [
            self.fpn_convs[idx](laterals[idx])
            for idx in range(used_feat_levels - 1)
        ]
        # add features from psp module
        fpn_outs.append(laterals[-1])

        # upsample feats from all level to the same size   
        for idx in range(used_feat_levels -1, 0, -1):
            fpn_outs[idx] = F.interpolate(
                fpn_outs[idx], 
                fpn_outs[0].shape[2:], 
                mode='bilinear', 
                align_corners=self.align_corners)
        fpn_outs = paddle.concat(fpn_outs, axis=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.conv_seg(output)
        output = F.interpolate(output, up4x_resolution, mode='bilinear', align_corners=self.align_corners)
        return output


