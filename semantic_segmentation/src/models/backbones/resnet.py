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

"""
Implement resnet50c backbone
"""

import os
import logging
import paddle
import paddle.nn as nn


class BasicBlockV1b(nn.Layer):
    """BasicBlockV1b Implement
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2D(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Layer):
    """BottleneckV1b Implement
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(nn.Layer):
    """ResNetV1
    """
    def __init__(self, block, layers, config, num_classes=1000, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2D):
        output_stride = config.MODEL.OUTPUT_STRIDE
        scale = config.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(ResNetV1, self).__init__()
        if deep_stem:
            # resnet vc
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, mid_channel, 3, 2, 1, bias_attr=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2D(mid_channel, mid_channel, 3, 1, 1, bias_attr=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2D(mid_channel, self.inplanes, 3, 1, 1, bias_attr=False)
            )
        else:
            self.conv1 = nn.Conv2D(3, self.inplanes, 7, 2, 3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2D(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2, norm_layer=norm_layer)

        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer, multi_grid=config.MODEL.ENCODER.MULTI_GRID,
                                       multi_dilation=config.MODEL.ENCODER.MULTI_DILATION)

        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape,
                                     dtype='float32', default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32',
                                     default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32',
                                     default_initializer=nn.initializer.Constant(value=0.0))
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, BottleneckV1b):
                    m.bn3.weight = paddle.create_parameter(shape=m.bn3.weight.shape,
                                             dtype='float32', default_initializer=nn.initializer.Constant(0.0))
                elif isinstance(m, BasicBlockV1b):
                    m.bn2.weight = paddle.create_parameter(shape=m.bn2.weight.shape,
                                             dtype='float32', default_initializer=nn.initializer.Constant(0.0))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2D,
                    multi_grid=False, multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride, bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div],
                                    previous_dilation=dilation, norm_layer=norm_layer))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation,
                                    previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


def resnet50c(config, norm_layer=nn.BatchNorm2D):
    """resnet50c implement
    
    The ResNet-50 [Heet al., 2016] with dilation convolution at last stage,
    ResNet-50 model Ref, https://arxiv.org/pdf/1512.03385.pdf

    Args:
        config (dict): configuration of network
        norm_layer: normalization layer type, default, nn.BatchNorm2D
    """
    num_block = [3, 4, 6, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer, deep_stem=True)


def load_backbone_pretrained(model, backbone, config):
    if config.MODEL.PRETRAINED:
        if os.path.isfile(config.MODEL.PRETRAINED):
            logging.info('Load pretrained backbone from local path!')
            model.set_state_dict(paddle.load(config.MODEL.PRETRAINED))


def get_segmentation_backbone(backbone, config, norm_layer=paddle.nn.BatchNorm2D):
    """
    Built the backbone model, defined by `config.MODEL.BACKBONE`.
    """
    model = resnet50c(config, norm_layer)
    load_backbone_pretrained(model, backbone, config)
    return model
