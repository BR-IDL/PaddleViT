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

import copy
import numpy as np
import math


class VisionTransformerUpHead(nn.Layer):
    """VisionTransformerUpHead

    VisionTransformerUpHead is the decoder of SETR-PUP, Ref https://arxiv.org/pdf/2012.15840.pdf

    Reference:                                                                                                                                                
        Sixiao Zheng, et al. *"Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"*
    """

    def __init__(self, embed_dim=1024, num_conv=1, num_upsample_layer=1, 
            conv3x3_conv1x1=True, align_corners=False, num_classes=60):
        super(VisionTransformerUpHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.num_conv = num_conv
        self.num_upsample_layer = num_upsample_layer
        self.conv3x3_conv1x1 = conv3x3_conv1x1

        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)) 
        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-06, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)

        if self.num_conv == 2:
            if self.conv3x3_conv1x1:
                self.conv_0 = nn.Conv2D(embed_dim, 256, 3, stride=1, padding=1, bias_attr=True)
            else:
                self.conv_0 = nn.Conv2D(embed_dim, 256, 1, stride=1, bias_attr=True)
            self.conv_1 = nn.Conv2D(256, self.num_classes, 1, stride=1)
            self.syncbn_fc_0 = nn.SyncBatchNorm(256, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)
        
        elif self.num_conv == 4:
            self.conv_0 = nn.Conv2D(embed_dim, 256, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_4 = nn.Conv2D(256, self.num_classes, kernel_size=1, stride=1)
            self.syncbn_fc_0 = nn.SyncBatchNorm(256, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)
            self.syncbn_fc_1 = nn.SyncBatchNorm(256, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)
            self.syncbn_fc_2 = nn.SyncBatchNorm(256, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)
            self.syncbn_fc_3 = nn.SyncBatchNorm(256, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr)

       

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def forward(self, x):
        x = self.norm(x)
        # (b,hw,c) -> (b,c,h,w)
        x = self.to_2D(x)
        up4x_resolution = [ 4*item for item in x.shape[2:]]
        up16x_resolution = [ 16*item for item in x.shape[2:]]
        if self.num_conv == 2:
            if self.num_upsample_layer == 2:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                x = F.interpolate(x, up4x_resolution, mode='bilinear', align_corners=self.align_corners)
                x = self.conv_1(x)
                x = F.interpolate(x, up16x_resolution, mode='bilinear', align_corners=self.align_corners)
            elif self.num_upsample_layer == 1:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                x = self.conv_1(x)
                x = F.interpolate(x, up16x_resolution, mode='bilinear', align_corners=self.align_corners)
        elif self.num_conv == 4:
            if self.num_upsample_layer == 4:
                x = self.conv_0(x)
                x = self.syncbn_fc_0(x)
                x = F.relu(x)
                up2x_resolution = [ 2*item for item in x.shape[2:]]
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
                x = self.conv_1(x)
                x = self.syncbn_fc_1(x)
                x = F.relu(x)
                up2x_resolution = [ 2*item for item in x.shape[2:]]
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
                x = self.conv_2(x)
                x = self.syncbn_fc_2(x)
                x = F.relu(x)
                up2x_resolution = [ 2*item for item in x.shape[2:]]
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
                x = self.conv_3(x)
                x = self.syncbn_fc_3(x)
                x = F.relu(x)
                x = self.conv_4(x)
                up2x_resolution = [ 2*item for item in x.shape[2:]]
                x = F.interpolate(x, up2x_resolution, mode='bilinear', align_corners=self.align_corners)
        return x


