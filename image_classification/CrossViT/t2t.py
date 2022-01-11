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
import math
import numpy as np
import paddle
import paddle.nn as nn
from crossvit_utils import DropPath, Identity, to_2tuple

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return paddle.to_tensor(sinusoid_table).unsqueeze(0)


class Token_performer(nn.Layer):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        # def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.0, dp2=0.0):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        w_attr_1, b_attr_1 = self._init_weights()
        self.kqv = nn.Linear(dim, 3 * self.emb, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.dp = nn.Dropout(dp1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(self.emb, self.emb, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.head_cnt = head_cnt
        w_attr_3, b_attr_3 = self._init_weights_norm()
        w_attr_4, b_attr_4 = self._init_weights_norm()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr_3, bias_attr=b_attr_3)
        self.norm2 = nn.LayerNorm(self.emb, weight_attr=w_attr_4, bias_attr=b_attr_4)
        self.epsilon = 1e-8  # for stable in division

        w_attr_5, b_attr_5 = self._init_weights()
        w_attr_6, b_attr_6 = self._init_weights()
        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb, weight_attr=w_attr_5, bias_attr=b_attr_5),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb, weight_attr=w_attr_6, bias_attr=b_attr_6),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = paddle.randn(self.m, self.emb)
        # todo wait implement
        # self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = paddle.matmul(x.float(), self.w, transpose_y=True)
        #wtx = paddlenlp.ops.einsum('bti,mi->btm', x.float(), self.w)

        return paddle.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = paddle.split(self.kqv(x), self.emb, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = paddle.matmul(qp, kp.sum(dim=1)).unsqueeze(dim=2)
        #D = paddlenlp.ops.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)
        kptv = paddle.matmul(v.float(), kp, transpose_x=True)
        #kptv = paddlenlp.ops.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = paddle.matmul(qp, kptv, transpose_y=True) / (D.repeat(1, 1, self.emb) + self.epsilon)
        #y = paddlenlp.ops.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)
        # skip connection
        y = v + self.dp(self.proj(y))

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features, hidden_features, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.act = act_layer()
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features, out_features, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.drop = nn.Dropout(drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(dim, in_dim * 3, weight_attr=w_attr_1, bias_attr=b_attr_1)

        self.attn_drop = nn.Dropout(attn_drop)
        w_attr_2, b_attr_2 = self._init_weights()
        self.proj = nn.Linear(in_dim, in_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x

        return x


class Token_transformer(nn.Layer):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights_norm()
        self.norm1 = norm_layer(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class T2T(nn.Layer):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        self.soft_split0 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[0][0]), strides=to_2tuple(kernel_size[0][1]),
                                     paddings=to_2tuple(kernel_size[0][2]))
        self.soft_split1 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[1][0]), strides=to_2tuple(kernel_size[1][1]),
                                     paddings=to_2tuple(kernel_size[1][2]))
        self.soft_split2 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[2][0]), strides=to_2tuple(kernel_size[2][1]),
                                     paddings=to_2tuple(kernel_size[2][2]))

        if tokens_type == 'transformer':

            self.attention1 = Token_transformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            w_attr_1, b_attr_1 = self._init_weights()
            self.project = nn.Linear(token_dim * (kernel_size[2][0] ** 2),
                                     embed_dim,
                                     weight_attr=w_attr_1,
                                     bias_attr=b_attr_1)

        elif tokens_type == 'performer':
            self.attention1 = Token_performer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            w_attr_1, b_attr_1 = self._init_weights()
            self.project = nn.Linear(token_dim * (kernel_size[2][0] ** 2),
                                     embed_dim,
                                     weight_attr=w_attr_1,
                                     bias_attr=b_attr_1)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * (img_size // (
                kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][
            1]))  # there are 3 sfot split, stride are 4,2,2 seperately

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class SharedT2T(nn.Layer):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        if tokens_type == 'transformer':
            # print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[0][0]),
                                         strides=to_2tuple(kernel_size[0][1]), paddings=to_2tuple(kernel_size[0][2]))
            self.soft_split1 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[1][0]),
                                         strides=to_2tuple(kernel_size[1][1]), paddings=to_2tuple(kernel_size[1][2]))
            self.soft_split2 = nn.Unfold(kernel_sizes=to_2tuple(kernel_size[2][0]),
                                         strides=to_2tuple(kernel_size[2][1]), paddings=to_2tuple(kernel_size[2][2]))

            self.attention1 = Token_transformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            w_attr_1, b_attr_1 = self._init_weights()
            self.project = nn.Linear(token_dim * (kernel_size[2][0] ** 2),
                                     embed_dim,
                                     weight_attr=w_attr_1,
                                     bias_attr=b_attr_1)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * (img_size // (
                kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1]))

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x
