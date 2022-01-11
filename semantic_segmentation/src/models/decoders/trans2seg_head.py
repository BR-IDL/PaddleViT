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
from ..backbones import expand


class ConvBNReLU(nn.Layer):
    '''ConvBNReLU
    
    Just contains Conv-BN-ReLU layer
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2D):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HybridEmbed(nn.Layer):
    """ CNN Feature Map Embedding
    
    Extract feature map from CNN, flatten, project to embedding dim.
    
    Attributes:
        input_dim: int, input dimension, default: 2048
        embed_dim: int, embedding dimension, default: 768
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


class SeparableConv2d(nn.Layer):
    '''Separable Conv2D
    
    Depthwise Separable Convolution, Ref, https://arxiv.org/pdf/1610.02357.pdf
    '''
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2D):
        super().__init__()
        depthwise = nn.Conv2D(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias_attr=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2D(inplanes, planes, 1, bias_attr=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(('relu', nn.ReLU()),
                                       ('depthwise', depthwise),
                                       ('bn_depth', bn_depth),
                                       ('pointwise', pointwise),
                                       ('bn_point', bn_point)
            )
        else:
            self.block = nn.Sequential(('depthwise', depthwise),
                                       ('bn_depth', bn_depth),
                                       ('relu1', nn.ReLU()),
                                       ('pointwise', pointwise),
                                       ('bn_point', bn_point),
                                       ('relu2', nn.ReLU())
            )

    def forward(self, x):
        return self.block(x)


class CNNHEAD(nn.Layer):
    """CNNHEAD Implement
    
    Attributes:
        vit_params: dict, input hyper params
        c1_channels: int, input channels, default, 256
        hid_dim: int, hidden dimension, default, 64
        norm_layer: normalization layer type, default: nn.BatchNorm2D
    """
    def __init__(self, vit_params, c1_channels=256, hid_dim=64, norm_layer=nn.BatchNorm2D):
        super().__init__()

        last_channels = vit_params['EMBED_DIM']
        nhead = vit_params['NUM_HEADS']
        self.conv_c1 = ConvBNReLU(c1_channels, hid_dim, 1, norm_layer=norm_layer)
        
        self.lay1 = SeparableConv2d(last_channels+nhead, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2D(hid_dim, 1, 1)
    
    def forward(self, x, c1, nclass, B):
        x = self.lay1(x)
        x = self.lay2(x)

        size = c1.shape[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.conv_c1(c1)
        x = x + expand(c1, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape([B, nclass, size[0], size[1]])
        
        return x
