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
Implement MobileFormer Arch
"""
import paddle
from paddle import nn

from dyrelu import DyReLU, MLP
from droppath import DropPath
from attention import Attention

class Stem(nn.Layer):
    """Stem
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.Hardswish):
        super(Stem, self).__init__(name_scope="Stem")
        conv_weight_attr, conv_bias_attr = self._conv_init()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              weight_attr=conv_weight_attr,
                              bias_attr=conv_bias_attr)
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = act()

    def _conv_init(self):
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
                 is_lite=False):
        super(DepthWiseConv, self).__init__(
                 name_scope="DepthWiseConv")
        self.is_lite = is_lite
        conv_weight_attr, conv_bias_attr = self._conv_init()
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
                # [[0, 1, 2]] -- [3, 1]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[kernel_size, 1],
                          stride=[stride, 1],
                          padding=[padding, 0],
                          groups=in_channels,
                        weight_attr=conv_weight_attr,
                        bias_attr=conv_bias_attr),
                nn.BatchNorm2D(in_channels),
                # [[0], [1], [2]] -- [1, 3]
                nn.Conv2D(in_channels=in_channels,
                          out_channels=in_channels,
                          kernel_size=[1, kernel_size],
                          stride=[1, stride],
                          padding=[0, padding],
                          groups=in_channels,
                        weight_attr=conv_weight_attr,
                        bias_attr=conv_bias_attr)
            )

    def _conv_init(self):
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
                 groups=1):
        super(PointWiseConv, self).__init__(
                 name_scope="PointWiseConv")
        conv_weight_attr, conv_bias_attr = self._conv_init()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              groups=groups,
                            weight_attr=conv_weight_attr,
                            bias_attr=conv_bias_attr)

    def _conv_init(self):
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
                 is_lite=False):
        super(BottleNeck, self).__init__(
                 name_scope="BottleNeck")
        self.is_lite = is_lite
        self.use_dyrelu = use_dyrelu
        assert use_dyrelu is False or (use_dyrelu is True and embed_dims is not None), \
               "Error: Please make sure while the use_dyrelu is True,"+\
               " embed_dims(now:{0})>0.".format(embed_dims)

        self.in_pw = PointWiseConv(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   groups=groups)
        self.in_pw_bn = nn.BatchNorm2D(hidden_channels)
        self.dw = DepthWiseConv(in_channels=hidden_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                is_lite=is_lite)
        self.dw_bn = nn.BatchNorm2D(hidden_channels)
        self.out_pw = PointWiseConv(in_channels=hidden_channels,
                                    out_channels=out_channels,
                                    groups=groups)
        self.out_pw_bn = nn.BatchNorm2D(out_channels)

        if use_dyrelu is False:
            self.act = nn.ReLU()
        else:
            self.act = DyReLU(in_channels=hidden_channels,
                                embed_dims=embed_dims,
                                k=k,
                                coefs=coefs,
                                consts=consts,
                                reduce=reduce)

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


class Classifier_Head(nn.Layer):
    """Classifier Head
        Params Info:
            in_channels: input feature map channels
            embed_dims: input token embed_dims
            hidden_features: the fc layer hidden feature size
            num_classes: the number of classes
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 hidden_features,
                 num_classes=1000,
                 act=nn.Hardswish):
        super(Classifier_Head, self).__init__(
                 name_scope="Classifier_Head")
        linear_weight_attr, linear_bias_attr = self._linear_init()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=in_channels+embed_dims,
                             out_features=hidden_features,
                             weight_attr=linear_weight_attr,
                             bias_attr=linear_bias_attr)
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=num_classes,
                             weight_attr=linear_weight_attr,
                             bias_attr=linear_bias_attr)
        self.act = act()
        self.softmax = nn.Softmax()

    def _linear_init(self):
        weight_attr = nn.initializer.KaimingNormal()
        bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def forward(self, feature_map, tokens):
        x = self.avg_pool(feature_map) # B, C, 1, 1
        x = self.flatten(x) # B, C

        z = tokens[:, 0] # B, 1, D
        x = paddle.concat([x, z], axis=-1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class Mobile(nn.Layer):
    """Mobile Sub-block
        Params Info:
            in_channels: input feature map channels
            hidden_channels: the dw layer hidden channel size
            groups: the number of groups, by 1x1conv
            embed_dims: input token embed_dims
            k: the number of parameters is in Dynamic ReLU
            coefs: the init value of coefficient parameters
            consts: the init value of constant parameters
            reduce: the mlp hidden scale,
                    means 1/reduce = mlp_ratio
            use_dyrelu: whether use dyrelu
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 embed_dims=None,
                 k=2,
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False):
        super(Mobile, self).__init__(
                 name_scope="Mobile")
        self.add_dw = True if stride==2 else False
        self.bneck = BottleNeck(in_channels=in_channels,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=1,
                                groups=groups,
                                embed_dims=embed_dims,
                                k=k,
                                coefs=coefs,
                                consts=consts,
                                reduce=reduce,
                                use_dyrelu=use_dyrelu)

        if self.add_dw: # stride==2
            self.downsample_dw = nn.Sequential(
                DepthWiseConv(in_channels=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding),
                nn.BatchNorm2D(in_channels)
                #, nn.ReLU()
            )

    def forward(self, feature_map, tokens):
        if self.add_dw:
            feature_map = self.downsample_dw(feature_map)

        x = self.bneck(feature_map, tokens)
        return x


class ToFormer_Bridge(nn.Layer):
    """Mobile to Former Bridge
        Params Info:
            in_channels: input feature map channels
            embed_dims: input token embed_dims
            num_head: the number of head is in multi head attention
            dropout_rate: the dropout rate of attention result
            attn_dropout_rate: the dropout rate of attention distribution
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(ToFormer_Bridge, self).__init__(
                 name_scope="ToFormer_Bridge")
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5
        linear_weight_attr, linear_bias_attr = self._linear_init()
        # split head to project
        self.heads_q_proj = []
        for i in range(num_head): # n linear
            self.heads_q_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims,
                          weight_attr=linear_weight_attr,
                          bias_attr=linear_bias_attr)
            )
        self.heads_q_proj = nn.LayerList(self.heads_q_proj)

        self.output = nn.Linear(in_features=self.num_head*self.head_dims,
                                out_features=embed_dims,
                                weight_attr=linear_weight_attr,
                                bias_attr=linear_bias_attr)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)

    def _linear_init(self):
        weight_attr = nn.initializer.KaimingNormal()
        bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def transfer_shape(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        assert C % self.num_head == 0, \
            "Erorr: Please make sure feature_map.channels % "+\
            "num_head == 0(now:{0}).".format(C % self.num_head)
        fm = feature_map.reshape(shape=[B, C, H*W]) # B, C, L
        fm = fm.transpose(perm=[0, 2, 1]) # B, L, C -- C = num_head * head_dims
        fm = fm.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        fm = fm.transpose(perm=[0, 2, 1, 3]) # B, n_h, L, h_d

        B, M, D = tokens.shape
        h_token = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        h_token = h_token.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h

        return fm, h_token

    def _multi_head_q_forward(self, token, B, M):
        q_list = []
        for i in range(self.num_head):
            q_list.append(
                # B, 1, M, head_dims
                self.heads_q_proj[i](token[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
        q = paddle.concat(q_list, axis=1) # B, num_head, M, head_dims
        return q

    def forward(self, feature_map, tokens):
        B, M, D = tokens.shape
        # fm（key/value） to shape: B, n_h, L, h_d
        # token to shape: B, n_h, M, D // n_h
        fm, token = self.transfer_shape(feature_map, tokens)

        q = self._multi_head_q_forward(token, B, M)

        # attention distribution
        attn = paddle.matmul(q, fm, transpose_y=True) # B, n_h, M, L
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # attention result
        z = paddle.matmul(attn, fm) # B, n_h, M, h_d
        z = z.transpose(perm=[0, 2, 1, 3])
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.output(z) # B, M, D
        z = self.dropout(z)
        z = z + tokens

        return z


class ToMobile_Bridge(nn.Layer):
    """Former to Mobile Bridge
        Params Info:
            in_channels: input feature map channels
            embed_dims: input token embed_dims
            num_head: the number of head is in multi head attention
            dropout_rate: the dropout rate of attention result
            attn_dropout_rate: the dropout rate of attention distribution
    """
    def __init__(self,
                 embed_dims,
                 in_channels,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.):
        super(ToMobile_Bridge, self).__init__(
                 name_scope="ToMobile_Bridge")
        self.num_head = num_head
        self.head_dims = in_channels // num_head
        self.scale = self.head_dims ** -0.5

        linear_weight_attr, linear_bias_attr = self._linear_init()

        self.heads_k_proj = []
        self.heads_v_proj = []
        for i in range(num_head): # n linear
            self.heads_k_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims,
                          weight_attr=linear_weight_attr,
                          bias_attr=linear_bias_attr)
            )
            self.heads_v_proj.append(
                nn.Linear(in_features=embed_dims // num_head,
                          out_features=self.head_dims,
                          weight_attr=linear_weight_attr,
                          bias_attr=linear_bias_attr)
            )
        self.heads_k_proj = nn.LayerList(self.heads_k_proj)
        self.heads_v_proj = nn.LayerList(self.heads_v_proj)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)

    def _linear_init(self):
        weight_attr = nn.initializer.KaimingNormal()
        bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def transfer_shape(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        assert C % self.num_head == 0, \
            "Erorr: Please make sure feature_map.channels % "+\
            "num_head == 0(now:{0}).".format(C % self.num_head)
        fm = feature_map.reshape(shape=[B, C, H*W]) # B, C, L
        fm = fm.transpose(perm=[0, 2, 1]) # B, L, C -- C = num_head * head_dims
        fm = fm.reshape(shape=[B, H*W, self.num_head, self.head_dims])
        fm = fm.transpose(perm=[0, 2, 1, 3]) # B, n_h, L, h_d

        B, M, D = tokens.shape
        k = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        k = k.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h
        v = tokens.reshape(shape=[B, M, self.num_head, D // self.num_head])
        v = v.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, D // n_h

        return fm, k, v

    def _multi_head_kv_forward(self, k_, v_, B, M):
        k_list = []
        v_list = []
        for i in range(self.num_head):
            k_list.append(
                # B, 1, M, head_dims
                self.heads_k_proj[i](k_[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
            v_list.append(
                # B, 1, M, head_dims
                self.heads_v_proj[i](v_[:, i, :, :]).reshape(
                                shape=[B, 1, M, self.head_dims])
            )
        k = paddle.concat(k_list, axis=1) # B, num_head, M, head_dims
        v = paddle.concat(v_list, axis=1) # B, num_head, M, head_dims
        return k, v

    def forward(self, feature_map, tokens):
        B, C, H, W = feature_map.shape
        B, M, D = tokens.shape

        # fm（q） to shape: B, n_h, L, h_d
        # k/v to shape: B, n_h, M, D // n_h
        q, k_, v_ = self.transfer_shape(feature_map, tokens)

        k, v = self._multi_head_kv_forward(k_, v_, B, M)

        # attention distribution
        attn = paddle.matmul(q, k, transpose_y=True) # B, n_h, L, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # attention result
        z = paddle.matmul(attn, v) # B, n_h, L, h_d
        z = z.transpose(perm=[0, 1, 3, 2]) # B, n_h, h_d, L
        # B, n_h*h_d, H, W
        z = z.reshape(shape=[B, self.num_head*self.head_dims, H, W])
        z = self.dropout(z)
        z = z + feature_map

        return z


class Former(nn.Layer):
    """Former Sub-block
        Params Info:
            embed_dims: input token embed_dims
            num_head: the number of head is in multi head attention
            mlp_ratio: the scale of hidden feature size
            dropout_rate: the dropout rate of attention result
            droppath_rate: the droppath rate of attention output
            attn_dropout_rate: the dropout rate of attention distribution
            mlp_dropout_rate: the dropout rate of mlp layer output
            qkv_bias: whether use the bias in qkv matrix
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 mlp_ratio=2,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 norm=nn.LayerNorm,
                 act=nn.GELU,
                 qkv_bias=True):
        super(Former, self).__init__(name_scope="Former")

        self.attn = Attention(embed_dims=embed_dims,
                                num_head=num_head,
                                dropout_rate=dropout_rate,
                                attn_dropout_rate=attn_dropout_rate,
                                qkv_bias=qkv_bias)
        self.attn_ln = norm(embed_dims)
        self.attn_droppath = DropPath(droppath_rate)

        self.mlp = MLP(in_features=embed_dims,
                        mlp_ratio=mlp_ratio,
                        mlp_dropout_rate=mlp_dropout_rate,
                        act=act)
        self.mlp_ln = norm(embed_dims)
        self.mlp_droppath = DropPath(droppath_rate)

    def forward(self, inputs):
        res = inputs
        x = self.attn(inputs)
        x = self.attn_ln(x)
        x = self.attn_droppath(x)
        x = x + res

        res = x
        x = self.mlp(x)
        x = self.mlp_ln(x)
        x = self.mlp_droppath(x)
        x = x + res

        return x


class MFBlock(nn.Layer):
    """MobileFormer Basic Block
        Params Info:
            in_channels: the number of input feature map channel
            hidden_channels: the number of hidden(dw_conv) feature map channel
            out_channels: the number of output feature map channel
            embed_dims: input token embed_dims
            num_head: the number of head is in multi head attention
            groups: the number of groups in 1x1 conv
            k: the number of parameters is in Dynamic ReLU
            coefs: the init value of coefficient parameters
            consts: the init value of constant parameters
            reduce: the mlp hidden scale,
                    means 1/reduce = mlp_ratio
            use_dyrelu: whether use dyrelu
            mlp_ratio: the scale of hidden feature size
            dropout_rate: the dropout rate of attention result
            droppath_rate: the droppath rate of attention output
            attn_dropout_rate: the dropout rate of attention distribution
            mlp_dropout_rate: the dropout rate of mlp layer output
            qkv_bias: whether use the bias in qkv matrix
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 k=2,
                 coefs=[1.0, 0.5],
                 consts=[1.0, 0.0],
                 reduce=4,
                 use_dyrelu=False,
                 num_head=1,
                 mlp_ratio=2,
                 dropout_rate=0.,
                 droppath_rate=0.,
                 attn_dropout_rate=0.,
                 mlp_dropout_rate=0.,
                 norm=nn.LayerNorm,
                 act=nn.GELU,
                 qkv_bias=True):
        super(MFBlock, self).__init__(
                 name_scope="MFBlock")
        self.mobile = Mobile(in_channels=in_channels,
                             hidden_channels=hidden_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             groups=groups,
                             embed_dims=embed_dims,
                             k=k,
                             coefs=coefs,
                             consts=consts,
                             reduce=reduce,
                             use_dyrelu=use_dyrelu)

        self.toformer_bridge = ToFormer_Bridge(embed_dims=embed_dims,
                                               in_channels=in_channels,
                                               num_head=num_head,
                                               dropout_rate=dropout_rate,
                                               attn_dropout_rate=attn_dropout_rate)
        self.toformer_norm = norm(embed_dims)

        self.former = Former(embed_dims=embed_dims,
                             num_head=num_head,
                             mlp_ratio=mlp_ratio,
                             dropout_rate=droppath_rate,
                             mlp_dropout_rate=mlp_dropout_rate,
                             attn_dropout_rate=attn_dropout_rate,
                             droppath_rate=droppath_rate,
                             norm=norm,
                             act=act)

        self.tomobile_bridge = ToMobile_Bridge(in_channels=out_channels,
                                               embed_dims=embed_dims,
                                               num_head=num_head,
                                               dropout_rate=dropout_rate,
                                               attn_dropout_rate=attn_dropout_rate)
        self.tomobile_norm = nn.BatchNorm2D(out_channels)


    def forward(self, feature_map, tokens):
        z_h = self.toformer_bridge(feature_map, tokens)
        z_h = self.toformer_norm(z_h)
        z_out = self.former(z_h)

        f_h = self.mobile(feature_map, z_out)
        f_out = self.tomobile_bridge(f_h, z_out)
        f_out = self.tomobile_norm(f_out)

        return f_out, z_out


class MobileFormer(nn.Layer):
    """MobileFormer
        Params Info:
            num_classes: the number of classes
            in_channels: the number of input feature map channel
            tokens: the shape of former token
            num_head: the number of head is in multi head attention
            groups: the number of groups in 1x1 conv
            k: the number of parameters is in Dynamic ReLU
            coefs: the init value of coefficient parameters
            consts: the init value of constant parameters
            reduce: the mlp hidden scale,
                    means 1/reduce = mlp_ratio
            use_dyrelu: whether use dyrelu
            mlp_ratio: the scale of hidden feature size
            dropout_rate: the dropout rate of attention result
            droppath_rate: the droppath rate of attention output
            attn_dropout_rate: the dropout rate of attention distribution
            mlp_dropout_rate: the dropout rate of mlp layer output
            alpha: the scale of model size
            qkv_bias: whether use the bias in qkv matrix
            config: total model config
    """
    def __init__(self, num_classes=1000, in_channels=3,
                 tokens=[3, 128], num_head=4, mlp_ratio=2,
                 use_dyrelu=True, k=2, reduce=4.0,
                 coefs=[1.0, 0.5], consts=[1.0, 0.0],
                 dropout_rate=0, droppath_rate=0,
                 attn_dropout_rate=0, mlp_dropout_rate=0,
                 norm=nn.LayerNorm, act=nn.GELU,
                 alpha=1.0, qkv_bias=True,
                 config=None):
        super(MobileFormer, self).__init__()
        self.num_token, self.embed_dims = tokens[0], tokens[1]
        self.num_head = num_head
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.mlp_ratio = mlp_ratio
        self.alpha = alpha
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.mlp_dropout_rate = mlp_dropout_rate

        assert config is not None, \
            "Error: Please enter the config(now: {0})".format(config)+\
            " in the __init__."

        # create learnable tokens: self.tokens
        self._create_token(num_token=self.num_token,
                           embed_dims=self.embed_dims)

        # create total model
        self._create_model(use_dyrelu=use_dyrelu,
                           reduce=reduce, dyrelu_k=k,
                           coefs=coefs, consts=consts,
                           alpha=alpha, norm=norm, act=act,
                           config=config)

    def _create_token(self, num_token, embed_dims):
        # B(1), token_size, embed_dims
        shape = [1] + [num_token, embed_dims]
        self.tokens = self.create_parameter(shape=shape, dtype='float32')

    def _create_stem(self,
                     in_channels,
                     out_channels,
                     kernel_size,
                     stride, padding,
                     alpha):
        self.stem = Stem(in_channels=in_channels,
                         out_channels=int(alpha * out_channels),
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)

    def _create_lite_bneck(self,
                           in_channels,
                           hidden_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           alpha,
                           pointwiseconv_groups):
        self.bneck_lite = BottleNeck(in_channels=int(alpha * in_channels),
                                     hidden_channels=int(alpha * hidden_channels),
                                     out_channels=int(alpha * out_channels),
                                     groups=pointwiseconv_groups,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     use_dyrelu=False,
                                     is_lite=True)

    def _create_mf_blocks(self,
                          in_channel_list,
                          hidden_channel_list,
                          out_channel_list,
                          kernel_list,
                          stride_list,
                          padding_list,
                          alpha,
                          use_dyrelu,
                          reduce,
                          dyrelu_k,
                          coefs,
                          consts,
                          norm,
                          act,
                          pointwiseconv_groups):
        self.blocks = []
        for i in range(0, len(in_channel_list)):
            self.blocks.append(
                MFBlock(
                    in_channels=int(alpha * in_channel_list[i]),
                    hidden_channels=int(alpha * hidden_channel_list[i]),
                    out_channels=int(alpha * out_channel_list[i]),
                    embed_dims=self.embed_dims,
                    kernel_size=kernel_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    groups=pointwiseconv_groups,
                    k=dyrelu_k,
                    coefs=coefs,
                    consts=consts,
                    reduce=reduce,
                    use_dyrelu=use_dyrelu,
                    num_head=self.num_head,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    droppath_rate=self.droppath_rate,
                    attn_dropout_rate=self.attn_dropout_rate,
                    mlp_dropout_rate=self.mlp_dropout_rate,
                    norm=norm,
                    act=act
                )
            )
        self.blocks = nn.LayerList(self.blocks)

    def _create_former_end_bridge(self,
                                  in_channels,
                                  norm,
                                  alpha):
        self.end_toformer_bridge = ToFormer_Bridge(embed_dims=self.embed_dims,
                                                    in_channels=int(alpha * in_channels),
                                                    num_head=self.num_head,
                                                    dropout_rate=self.dropout_rate,
                                                    attn_dropout_rate=self.attn_dropout_rate)
        self.former_bridge_norm = norm(self.embed_dims)

    def _create_channel_conv(self,
                             in_channels,
                             out_channels,
                             alpha,
                             pointwiseconv_groups):
        self.channel_conv = nn.Sequential(
            PointWiseConv(in_channels=int(alpha * in_channels),
                          out_channels=out_channels,
                          groups=pointwiseconv_groups),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def _create_head(self,
                     in_channels,
                     hidden_features):
        self.head = Classifier_Head(in_channels=in_channels,
                                    embed_dims=self.embed_dims,
                                    hidden_features=hidden_features,
                                    num_classes=self.num_classes)

    def _create_model(self,
                      use_dyrelu,
                      reduce,
                      dyrelu_k,
                      coefs,
                      consts,
                      norm,
                      act,
                      alpha,
                      config):
        # create stem: self.stem
        self._create_stem(in_channels=self.in_channels,
                          out_channels=config.MODEL.MF.STEM.OUT_CHANNELS,
                          kernel_size=config.MODEL.MF.STEM.KERNELS,
                          stride=config.MODEL.MF.STEM.STRIEDS,
                          padding=config.MODEL.MF.STEM.PADDINGS,
                          alpha=alpha)
        # create lite-bottleneck: self.bneck_lite
        self._create_lite_bneck(in_channels=config.MODEL.MF.LITE_BNECK.IN_CHANNEL,
                                hidden_channels=config.MODEL.MF.LITE_BNECK.HIDDEN_CHANNEL,
                                out_channels=config.MODEL.MF.LITE_BNECK.OUT_CHANNEL,
                                kernel_size=config.MODEL.MF.LITE_BNECK.KERNEL,
                                stride=config.MODEL.MF.LITE_BNECK.STRIED,
                                padding=config.MODEL.MF.LITE_BNECK.PADDING,
                                alpha=alpha,
                                pointwiseconv_groups=config.MODEL.MF.POINTWISECONV_GROUPS)
        # create mobileformer blocks: self.blocks
        self._create_mf_blocks(in_channel_list=config.MODEL.MF.BLOCK.IN_CHANNELS,
                               hidden_channel_list=config.MODEL.MF.BLOCK.HIDDEN_CHANNELS,
                               out_channel_list=config.MODEL.MF.BLOCK.OUT_CHANNELS,
                               kernel_list=config.MODEL.MF.BLOCK.KERNELS,
                               stride_list=config.MODEL.MF.BLOCK.STRIEDS,
                               padding_list=config.MODEL.MF.BLOCK.PADDINGS,
                               alpha=alpha,
                               use_dyrelu=use_dyrelu,
                               reduce=reduce,
                               dyrelu_k=dyrelu_k,
                               coefs=coefs,
                               consts=consts,
                               norm=norm,
                               act=act,
                               pointwiseconv_groups=config.MODEL.MF.POINTWISECONV_GROUPS)
        # create final toformer_bridge: self.toformer_bridge
        self._create_former_end_bridge(in_channels=config.MODEL.MF.CHANNEL_CONV.IN_CHANNEL,
                                       norm=norm,
                                       alpha=alpha)
        # create channel 1x1 conv: self.channel_conv
        self._create_channel_conv(in_channels=config.MODEL.MF.CHANNEL_CONV.IN_CHANNEL,
                                  out_channels=config.MODEL.MF.CHANNEL_CONV.OUT_CHANNEL,
                                  alpha=alpha,
                                  pointwiseconv_groups=config.MODEL.MF.POINTWISECONV_GROUPS)
        # create classifier head: self.head
        self._create_head(in_channels=config.MODEL.MF.HEAD.IN_CHANNEL,
                          hidden_features=config.MODEL.MF.HEAD.HIDDEN_FEATURE)

    def _to_batch_tokens(self, batch_size):
        # B, token_size, embed_dims
        return paddle.concat([self.tokens]*batch_size,  axis=0)

    def bridge_forward(self, inputs):
        B, _, _, _ = inputs.shape
        feature_map = self.stem(inputs)
        # create batch tokens
        tokens = self._to_batch_tokens(B) # B, token_size, embed_dims
        feature_map = self.bneck_lite(feature_map, tokens)

        for b in self.blocks:
            feature_map, tokens = b(feature_map, tokens)

        tokens = self.end_toformer_bridge(feature_map, tokens)
        tokens = self.former_bridge_norm(tokens)

        return feature_map, tokens

    def forward(self, inputs):
        feature_map, tokens = self.bridge_forward(inputs)

        feature_map = self.channel_conv(feature_map)
        output = self.head(feature_map, tokens)

        return output


def build_mformer(config):
    """build model
    """
    model = MobileFormer(num_classes=config.MODEL.NUM_CLASSES,
                         in_channels=config.MODEL.MF.IN_CHANNELS,
                         tokens=config.MODEL.MF.TOKENS,
                         num_head=config.MODEL.MF.NUM_HEAD,
                         mlp_ratio=config.MODEL.MF.MLP_RATIO,
                         k=config.MODEL.MF.DYRELU.DYRELU_K,
                         reduce=config.MODEL.MF.DYRELU.REDUCE,
                         coefs=config.MODEL.MF.DYRELU.COEFS,
                         consts=config.MODEL.MF.DYRELU.CONSTS,
                         use_dyrelu=config.MODEL.MF.DYRELU.USE_DYRELU,
                         dropout_rate=config.MODEL.DROPOUT,
                         droppath_rate=config.MODEL.DROPPATH,
                         attn_dropout_rate=config.MODEL.ATTENTION_DROPOUT,
                         mlp_dropout_rate=config.MODEL.MLP_DROPOUT,
                         alpha=config.MODEL.MF.ALPHA,
                         qkv_bias=config.MODEL.MF.QKV_BIAS,
                         config=config)

    return model
