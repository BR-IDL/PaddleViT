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

import copy
import paddle
import paddle.nn as nn


def readout_oper(config):
    """get the layer to process the feature asnd the cls token
    """
    class Drop(object):
        """drop class
        just drop the cls token
        """
        def __init__(self, config):
            if 'ViT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 1
            elif 'DeiT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 2
            self.feature_size = (config.DATA.CROP_SIZE[0] // config.MODEL.TRANS.PATCH_SIZE,
                                config.DATA.CROP_SIZE[1] // config.MODEL.TRANS.PATCH_SIZE)

        def __call__(self, x):
            x = x[:, self.token_num:]
            x = x.transpose((0, 2, 1))
            x = x.reshape((x.shape[0], x.shape[1], self.feature_size[0], self.feature_size[1]))
            return x


    class Add(object):
        """add class
        add the cls token
        """
        def __init__(self, config):
            if 'ViT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 1
            elif 'DeiT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 2
            self.feature_size = (config.DATA.CROP_SIZE[0] // config.MODEL.TRANS.PATCH_SIZE,
                                config.DATA.CROP_SIZE[1] // config.MODEL.TRANS.PATCH_SIZE)

        def __call__(self, x):
            token = x[:, :self.token_num]
            token = paddle.sum(token, axis=1).unsqueeze(1)
            x = x[:, self.token_num:]
            x = x + token
            x = x.transpose((0, 2, 1))
            x = x.reshape((x.shape[0], x.shape[1], self.feature_size[0], self.feature_size[1]))
            return x

    class Proj(nn.Layer):
        """porject class
        use a linear layer to confuse the feature and the cls token
        """
        def __init__(self, config):
            super(Proj, self).__init__()
            if 'ViT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 1
            elif 'DeiT' in config.MODEL.ENCODER.TYPE:
                self.token_num = 2
            self.feature_size = (config.DATA.CROP_SIZE[0] // config.MODEL.TRANS.PATCH_SIZE,
                                config.DATA.CROP_SIZE[1] // config.MODEL.TRANS.PATCH_SIZE)
            self.proj = nn.Sequential(
                nn.Linear(2 * config.MODEL.TRANS.HIDDEN_SIZE, config.MODEL.TRANS.HIDDEN_SIZE),
                nn.GELU()
            )

        def forward(self, x):
            token = x[:, :self.token_num]
            token = paddle.sum(token, axis=1).unsqueeze(1)
            x = x[:, self.token_num:]
            token = token.expand_as(x)
            x = paddle.concat([x, token], axis=-1)
            x = self.proj(x)
            x = x.transpose((0, 2, 1))
            x = x.reshape((x.shape[0], x.shape[1], self.feature_size[0], self.feature_size[1]))
            return x

    if config.MODEL.DPT.READOUT_PROCESS == 'drop':
        return [copy.deepcopy(Drop(config)) for _ in range(4)]
    if config.MODEL.DPT.READOUT_PROCESS == 'add':
        return [copy.deepcopy(Add(config)) for _ in range(4)]
    if config.MODEL.DPT.READOUT_PROCESS =='project':
        return nn.LayerList([copy.deepcopy(Proj(config)) for _ in range(4)])
    return None

class ResidualBLock(nn.Layer):
    """Residual block
    """
    def __init__(self, channels, bn=True, act=nn.ReLU):
        super(ResidualBLock, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2D(channels, channels, 3, 1, 1, bias_attr=not self.bn)
        self.conv2 = nn.Conv2D(channels, channels, 3, 1, 1, bias_attr=not self.bn)
        if bn:
            self.bn1 = nn.BatchNorm2D(channels)
            self.bn2 = nn.BatchNorm2D(channels)
        self.act=act()

    def forward(self, inputs):
        x = self.act(inputs)
        x = self.conv1(x)
        if self.bn:
            x=self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        return inputs+x


class FeatureFusionBlock(nn.Layer):
    """Feature fusion block
    """
    def __init__(self, channels, act, bn=True, expand=True, align_corners=True, outconv=True):
        super(FeatureFusionBlock, self).__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_channels = channels // 2 if expand else channels

        self.out_conv = nn.Conv2D(channels, out_channels, 1, 1, 0, bias_attr=True)if outconv else None
        self.resblock1 = ResidualBLock(channels, bn, act)
        self.resblock2 = ResidualBLock(channels, bn, act)


    def forward(self, feature, x):
        if x is not None:
            x += self.resblock1(feature)
        else:
            x = feature
        x = self.resblock2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if self.out_conv:
            return self.out_conv(x)
        return x


class DPTHead(nn.Layer):
    """DPTHead
     
    DPTHead is the decoder of the dense predition transformer (DPT), Ref https://arxiv.org/pdf/2103.13413.pdf.
    Reference:                                                                                                                                                
        Rene Ranftl, et al. *"Vision Transformers for Dense Prediction"*
    """
    def __init__(self, config):
        super(DPTHead, self).__init__()
        features=config.MODEL.DPT.FEATURES
        self.readout_oper = readout_oper(config)
        self.refine = nn.LayerList([
            copy.deepcopy(FeatureFusionBlock(
                channels=features,
                act=nn.ReLU,
                bn=True,
                expand=False,
                align_corners=True
            ))for _ in range(4)
        ])
        self.layers_rn = get_scratch(config)
        self.process = get_process(config)
        self.head = nn.Sequential(
            nn.Conv2D(features, features, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(features),
            nn.ReLU(),
            nn.Dropout2D(0.1),
            nn.Conv2D(features, config.DATA.NUM_CLASSES, 1),
        )


    def forward(self, inputs):
        x = None
        for i in range(3, -1, -1):
            feature = self.readout_oper[i](inputs[i])
            feature = self.process[i](feature)
            feature = self.layers_rn[i](feature)
            x = self.refine[i](feature, x)
        x = self.head(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return [x,]


def get_scratch(config, groups=1, expand=False):
    """function to get the layer to make sure the features have the same dims
    """
    if expand:
        out_shape = [config.MODEL.DPT.FEATURES * 2 ** i for i in range(4)]
    else:
        out_shape = [config.MODEL.DPT.FEATURES for _ in range(4)]
    layers_rn=nn.LayerList()
    for i in range(4):
        layers_rn.append(nn.Conv2D(
            config.MODEL.DPT.HIDDEN_FEATURES[i],
            out_shape[i],
            kernel_size=3,
            stride=1,
            padding='same',
            bias_attr=False,
            groups=groups
        ))
    return layers_rn

def get_process(config):
    """
    function to get the layers to process the feature from the backbone
    """
    process = nn.LayerList()
    process.append(
        nn.Sequential(
            nn.Conv2D(
                in_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[0],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Conv2DTranspose(
                in_channels=config.MODEL.DPT.HIDDEN_FEATURES[0],
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias_attr=True,
                dilation=1,
                groups=1
            )
        )
    )
    process.append(
        nn.Sequential(
            nn.Conv2D(
                in_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[1],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Conv2DTranspose(
                in_channels=config.MODEL.DPT.HIDDEN_FEATURES[1],
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias_attr=True,
                dilation=1,
                groups=1
            )
        )
    )
    process.append(
        nn.Sequential(
            nn.Conv2D(
                in_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[2],
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
    )
    process.append(
        nn.Sequential(
            nn.Conv2D(
                in_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[3],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Conv2D(
                in_channels=config.MODEL.DPT.HIDDEN_FEATURES[3],
                out_channels=config.MODEL.DPT.HIDDEN_FEATURES[3],
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=True,
                dilation=1,
                groups=1
            )
        )
    )
    return process
