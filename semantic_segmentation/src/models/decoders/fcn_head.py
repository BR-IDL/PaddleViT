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

class FCNHead(nn.Layer):
    """FCNHead

    FCNHead is the decoder of FCN, which can also be used as an auxiliary segmentation head for other segmentation models.
    Ref https://arxiv.org/pdf/1411.4038.pdf
    Reference:                                                                                                                                                
        Jonathan Long, et al. *"Fully Convolution Networks for Semantic Segmentation."*
    """
    def __init__(self, 
                 in_channels=384, 
                 channels=256, 
                 num_convs=1, 
                 concat_input=False, 
                 dropout_ratio=0.1, 
                 num_classes=60, 
                 up_ratio = 16,
                 align_corners=False):
        super(FCNHead, self).__init__()   
        self.in_channels = in_channels
        self.channels = channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.up_ratio = up_ratio
        self.align_corners = align_corners
        if num_convs ==0:
            assert self.in_channels == self.channels
        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)) 
        convs = []
        for i in range(self.num_convs):
            in_channels = self.in_channels if i==0 else self.channels 
            conv = nn.Sequential(
                nn.Conv2D(in_channels, self.channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat=nn.Sequential(
                nn.Conv2D(self.in_channels+self.channels, self.channels, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())
        if dropout_ratio > 0:
            self.dropout= nn.Dropout2D(p=dropout_ratio)
        self.conv_seg = nn.Conv2D(self.channels, self.num_classes, kernel_size=1)
       

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, x):
        up_resolution = [ self.up_ratio*item for item in x.shape[2:]]
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(paddle.concat([x, output], axis=1))
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        output = F.interpolate(output, up_resolution, mode='bilinear', align_corners=self.align_corners)
        return output
            

            
