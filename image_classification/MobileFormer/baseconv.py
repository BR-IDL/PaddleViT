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
"""
MobileFormer Arch -- Base Conv Type Implement
"""
import paddle
from paddle import nn

from dyrelu import DyReLU

class Stem(nn.Layer):
    """Stem
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish,
                 init_type='kn'):
        super(Stem, self).__init__(
                 name_scope="Stem")
        conv_weight_attr, conv_bias_attr = self._conv_init(init_type=init_type)
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              weight_attr=conv_weight_attr,
                              bias_attr=conv_bias_attr)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()

    def _conv_init(self, init_type='kn'):
        if init_type == 'xu':
            weight_attr = nn.initializer.XavierUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'ku':
            weight_attr = nn.initializer.KaimingUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'kn':
            weight_attr = nn.initializer.KaimingNormal()
            bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthWiseConv(nn.Layer):
    """DepthWise Conv -- support lite weight dw_conv
        Params Info:
            is_lite: use lite weight dw_conv
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 is_lite=False,
                 init_type='kn'):
        super(DepthWiseConv, self).__init__(
                 name_scope="DepthWiseConv")
        self.is_lite = is_lite

        conv_weight_attr, conv_bias_attr = self._conv_init(init_type=init_type)
        if is_lite is False:
            self.conv = nn.Conv2D(in_channels=in_channels,
                                out_channels=in_channels,
                                groups=in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                weight_attr=conv_weight_attr,
                                bias_attr=conv_bias_attr)
        else:
            self.conv = nn.Sequential(
                # kernel_size -- [3, 1]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[kernel_size, 1],
                          stride=[stride, 1],
                          padding=[padding, 0],
                          groups=in_channels,
                        weight_attr=conv_weight_attr,
                        bias_attr=conv_bias_attr),
                nn.BatchNorm2D(in_channels),
                # kernel_size -- [1, 3]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[1, kernel_size],
                          stride=[1, stride],
                          padding=[0, padding],
                          groups=in_channels,
                        weight_attr=conv_weight_attr,
                        bias_attr=conv_bias_attr)
            )

    def _conv_init(self, init_type='kn'):
        if init_type == 'xu':
            weight_attr = nn.initializer.XavierUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'ku':
            weight_attr = nn.initializer.KaimingUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'kn':
            weight_attr = nn.initializer.KaimingNormal()
            bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class PointWiseConv(nn.Layer):
    """PointWise 1x1Conv -- support group conv
        Params Info:
            groups: the number of groups
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 init_type='kn'):
        super(PointWiseConv, self).__init__(
                 name_scope="PointWiseConv")
        conv_weight_attr, conv_bias_attr = self._conv_init(init_type=init_type)
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=groups,
                            weight_attr=conv_weight_attr,
                            bias_attr=conv_bias_attr)

    def _conv_init(self, init_type='kn'):
        if init_type == 'xu':
            weight_attr = nn.initializer.XavierUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'ku':
            weight_attr = nn.initializer.KaimingUniform()
            bias_attr = nn.initializer.Constant(value=0.0)
        elif init_type == 'kn':
            weight_attr = nn.initializer.KaimingNormal()
            bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class BottleNeck(nn.Layer):
    """BottleNeck
        Params Info:
            groups: the number of groups, by 1x1conv
            embed_dims: input token embed_dims
            k: the number of parameters is in Dynamic ReLU
            coefs: the init value of coefficient parameters
            consts: the init value of constant parameters
            reduce: the mlp hidden scale,
                    means 1/reduce = mlp_ratio
            use_dyrelu: whether use dyrelu
            is_lite: whether use lite dw_conv
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 groups=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 embed_dims=None,
                 k=2, # the number of dyrelu-params
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False,
                 is_lite=False,
                 init_type='kn'):
        super(BottleNeck, self).__init__(
                 name_scope="BottleNeck")
        self.is_lite = is_lite
        self.use_dyrelu = use_dyrelu

        assert use_dyrelu==False or (use_dyrelu==True and embed_dims is not None), \
               "Error: Please make sure while the use_dyrelu==True,"+\
               " embed_dims(now:{0})>0.".format(embed_dims)

        self.in_pw = PointWiseConv(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   groups=groups,
                                   init_type=init_type)
        self.in_pw_bn = nn.BatchNorm2D(hidden_channels)

        self.dw = DepthWiseConv(in_channels=hidden_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                is_lite=is_lite,
                                init_type=init_type)
        self.dw_bn = nn.BatchNorm2D(hidden_channels)

        self.out_pw = PointWiseConv(in_channels=hidden_channels,
                                    out_channels=out_channels,
                                    groups=groups,
                                    init_type=init_type)
        self.out_pw_bn = nn.BatchNorm2D(out_channels)

        if use_dyrelu == False:
            self.act = nn.ReLU()
        else:
            self.act = DyReLU(in_channels=hidden_channels,
                                embed_dims=embed_dims,
                                k=k,
                                coefs=coefs,
                                consts=consts,
                                reduce=reduce,
                                init_type=init_type)

    def forward(self, feature_map, tokens):
        x = self.in_pw(feature_map)
        x = self.in_pw_bn(x)
        if self.use_dyrelu:
            x = self.act(x, tokens)

        x = self.dw(x)
        x = self.dw_bn(x)
        if self.use_dyrelu:
            x = self.act(x, tokens)

        x = self.out_pw(x)
        x = self.out_pw_bn(x)

        return x