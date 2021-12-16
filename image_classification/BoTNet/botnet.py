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

'''
Implement BoTNet
'''

import paddle
import paddle.nn as nn
from resnet import resnet50
import numpy as np


def expand_dim(t, dim, k):
    """

    Expand dims for t at dim to k

    """
    t = t.unsqueeze(axis=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return paddle.expand(t, expand_shape)


def rel_to_abs(x):
    """

    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = paddle.zeros([B, Nh, L, 1])
    x = paddle.concat([x, col_pad], axis=3)
    flat_x = x.reshape([B, Nh, L * 2 * L])
    flat_pad = paddle.zeros([B, Nh, L - 1])
    flat_x = paddle.concat([flat_x, flat_pad], axis=2)
    # Reshape and slice out the padded elements
    final_x = flat_x.reshape([B, Nh, L + 1, 2 * L - 1])
    return final_x[:, :, :L, L - 1 :]


def relative_logits_1d(q, rel_k):
    """

    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.

    """
    B, Nh, H, W, _ = q.shape
    rel_logits = paddle.matmul(q, rel_k.T)
    # Collapse height and heads
    rel_logits = rel_logits.reshape([-1, Nh * H, W, 2 * W - 1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = rel_logits.reshape([-1, Nh, H, W, W])
    rel_logits = expand_dim(rel_logits, dim=3, k=H)
    return rel_logits


class RelPosEmb(nn.Layer):
    def __init__(self, 
                height, 
                width, 
                dim_head):
        super().__init__()

        scale = dim_head ** -0.5
        self.height = height
        self.width = width
        h_shape = [height * 2 - 1, dim_head]
        w_shape = [width * 2 - 1, dim_head]
        self.rel_height = paddle.create_parameter(
            shape=h_shape, dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(h_shape)*scale)
        )
        self.rel_width = paddle.create_parameter(
            shape=w_shape, dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(w_shape)*scale)
        )

    def forward(self, q):

        H = self.height
        W = self.width
        B, N, _, D = q.shape
        q = q.reshape([B, N, H, W, D]) # "B N (H W) D -> B N H W D"
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.transpose(perm=[0, 1, 2, 4, 3, 5])
        B, N, X, I, Y, J = rel_logits_w.shape
        rel_logits_w = rel_logits_w.reshape([B, N, X*Y, I*J]) # "B N X I Y J-> B N (X Y) (I J)"

        q = q.transpose(perm=[0, 1, 3, 2, 4]) # "B N H W D -> B N W H D"
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.transpose(perm=[0, 1, 4, 2, 5, 3])
        B, N, X, I ,Y, J = rel_logits_h.shape
        rel_logits_h = rel_logits_h.reshape([B, N, Y*X, J*I]) # "B N X I Y J -> B N (Y X) (J I)"

        return rel_logits_w + rel_logits_h


class BoTBlock(nn.Layer):
    def __init__(self,
                dim,
                fmap_size,
                dim_out,
                stride=1,
                heads=4,
                proj_factor=4,
                dim_qk=128,
                dim_v=128,
                rel_pos_emb=False,
                activation=nn.ReLU()):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()

        if dim != dim_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2D(dim, dim_out, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(dim_out),
                activation,
            )
        else:
            self.shortcut = nn.Identity()

        bottleneck_dimension = dim_out // proj_factor
        attn_dim_out = heads * dim_v

        self.net = nn.Sequential(
            nn.Conv2D(dim, bottleneck_dimension, kernel_size=1, stride=1, bias_attr=False),
            nn.BatchNorm2D(bottleneck_dimension),
            activation,
            MHSA(
                dim=bottleneck_dimension,
                fmap_size=fmap_size,
                heads=heads,
                dim_qk=dim_qk,
                dim_v=dim_v,
                rel_pos_emb=rel_pos_emb,
            ),
            nn.AvgPool2D(2) if stride == 2 else nn.Identity(),
            nn.BatchNorm2D(attn_dim_out),
            activation,
            nn.Conv2D(attn_dim_out, dim_out, kernel_size=1, stride=1, bias_attr=False),
            nn.BatchNorm2D(dim_out),
        )

        self.activation = activation

    def forward(self, featuremap):
        shortcut = self.shortcut(featuremap)
        featuremap = self.net(featuremap)
        featuremap += shortcut
        return self.activation(featuremap)


class MHSA(nn.Layer):
    '''Multi-Head Self-Attention'''
    def __init__(self, 
                dim, 
                fmap_size, 
                heads=4, 
                dim_qk=128, 
                dim_v=128, 
                rel_pos_emb=False):
        """
        dim: number of channels of feature map
        fmap_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()

        self.scale = dim_qk ** -0.5
        self.heads = heads
        out_channels_qk = heads * dim_qk
        out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2D(dim, out_channels_qk * 2, 1, bias_attr=False)
        self.to_v = nn.Conv2D(dim, out_channels_v, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=-1)

        height, width = fmap_size
        self.pos_emb = RelPosEmb(height, width, dim_qk)

    def transpose_multihead(self, x):
        B, N, H, W = x.shape
        x = x.reshape([B, self.heads, -1, H, W]) # "B (h D) H W -> B h D H W"
        x = x.transpose(perm=[0, 1, 3, 4, 2])    # "B h D H W -> B h H W D"
        x = x.reshape([B, self.heads, H*W, -1])  # "B h H W D -> B h (H W) D"
        return x

    def forward(self, featuremap):
        """
        featuremap: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        B, C, H, W = featuremap.shape
        q, k = self.to_qk(featuremap).chunk(2, axis=1)
        v = self.to_v(featuremap)
        q, k, v = map(self.transpose_multihead, [q, k, v])
        q *= self.scale

        logits = paddle.matmul(q, k.transpose(perm=[0, 1, 3, 2]))
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = paddle.matmul(weights, v)
        a_B, a_N, a_, a_D = attn_out.shape
        attn_out = attn_out.reshape([a_B, a_N, H, -1, a_D])  # "B N (H W) D -> B N H W D"
        attn_out = attn_out.transpose(perm=[0, 1, 4, 2, 3])  # "B N H W D -> B N D H W"
        attn_out = attn_out.reshape([a_B, a_N*a_D, H, -1]) # "B N D H W -> B (N D) H W"
        return attn_out


class BoTStack(nn.Layer):
    def __init__(self,
                dim,
                fmap_size,
                dim_out=2048,
                heads=4,
                proj_factor=4,
                num_layers=3,
                stride=2,
                dim_qk=128,
                dim_v=128,
                rel_pos_emb=False,
                activation=nn.ReLU(),):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out

            fmap_divisor = 2 if stride == 2 and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BoTBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    stride=stride if is_first else 1,
                    heads=heads,
                    proj_factor=proj_factor,
                    dim_qk=dim_qk,
                    dim_v=dim_v,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f"assert {c} == self.dim {self.dim}"
        assert h == self.fmap_size[0] and w == self.fmap_size[1]
        return self.net(x)


def botnet50(pretrained=False, 
            image_size=224, 
            fmap_size=(14, 14), 
            num_classes=1000, 
            embed_dim=2048,
            **kwargs):
    """
    Bottleneck Transformers for Visual Recognition.
    """
    resnet = resnet50(pretrained=False, **kwargs)
    layer = BoTStack(dim=1024, dim_out=embed_dim, fmap_size=fmap_size, stride=1, rel_pos_emb=True)
    backbone = list(resnet.children())
    model = nn.Sequential(
        *backbone[:-3],
        layer,
        nn.AdaptiveAvgPool2D([1, 1]),
        nn.Flatten(1),
        nn.Linear(embed_dim, num_classes),
    )
    if pretrained:
        state_dict = paddle.load('botnet50.pdparams')
        model.set_state_dict(state_dict)
    return model


def build_botnet50(config):
    model = botnet50(
        image_size=config.DATA.IMAGE_SIZE,
        fmap_size=config.DATA.FMAP_SIZE,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.TRANS.EMBED_DIM,
    )
    return model