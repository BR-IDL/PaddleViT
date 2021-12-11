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
Implement Transformer Class for ViT
"""

import copy
import paddle
import paddle.nn as nn
from droppath import DropPath


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PatchEmbedding(nn.Layer):
    """Patch Embedding and Position Embedding
    Apply patch embedding and position embedding on input images.
    Attributes:
        patch_embddings: impl using a patch_size x patch_size Conv2D operation
        position_embddings: a parameter with len = num_patch + 1(for cls_token)
        cls_token: token insert to the patch feature for classification
        dropout: dropout for embeddings
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)

        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

        self.position_embeddings = paddle.create_parameter(
            shape=[1, n_patches + 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat((cls_tokens, x), axis=1)

        embeddings = x + self.position_embeddings  # tensor broadcast
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Layer):
    """ Attention module
    Attention module for ViT, here q, k, v are assumed the same.
    The qkv mappings are stored as one single param.
    Attributes:
        num_heads: number of heads
        attn_head_size: feature dim of single head
        all_head_size: feature dim of all heads
        qkv: a nn.Linear for q, k, v mapping
        scales: 1 / sqrt(single_head_feature_dim)
        out: projection of multi-head attention
        attn_dropout: dropout for attention
        proj_dropout: final dropout before output
        softmax: softmax op for attention
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()

        assert isinstance(embed_dim, int), (
            f"Expected the type of `embed_dim` to be {int}, but received {type(embed_dim)}.")
        assert isinstance(num_heads, int), (
            f"Expected the type of `num_heads` to be {int}, but received {type(num_heads)}.")

        assert embed_dim > 0, (
            f"Expected `embed_dim` to be greater than 0, but received {embed_dim}")
        assert num_heads > 0, (
            f"Expected `num_heads` to be greater than 0, but received {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if attn_head_size is not None:
            assert isinstance(attn_head_size, int), (
                f"Expected the type of `attn_head_size` to be {int}, "
                f"but received {type(attn_head_size)}.")
            assert attn_head_size > 0, f"Expected `attn_head_size` to be greater than 0," \
                                       f" but received {attn_head_size}."
            self.attn_head_size = attn_head_size
        else:
            self.attn_head_size = embed_dim // num_heads
            assert self.attn_head_size * num_heads == embed_dim, (
                f"`embed_dim` must be divisible by `num_heads`,"
                f" but received embed_dim={embed_dim}, num_heads={num_heads}.")

        self.all_head_size = self.attn_head_size * num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3,  # weights for q, k, and v
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(self.all_head_size,
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.attn_head_size]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)
        # reshape
        z = self.out(z)
        z = self.proj_dropout(z)
        return z


class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """

    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=0.2))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderLayer(nn.Layer):
    """Encoder Layer
    Encoder layer contains attention, norm, mlp and residual
    Attributes:
        hidden_size: transformer feature dim
        attn_norm: nn.LayerNorm before attention
        mlp_norm: nn.LayerNorm before mlp
        mlp: mlp modual
        attn: attention modual
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.attn_norm = nn.LayerNorm(embed_dim,
                                      weight_attr=w_attr_1,
                                      bias_attr=b_attr_1,
                                      epsilon=1e-6)

        self.attn = Attention(embed_dim,
                              num_heads,
                              attn_head_size,
                              qkv_bias,
                              dropout,
                              attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()

        w_attr_2, b_attr_2 = self._init_weights()
        self.mlp_norm = nn.LayerNorm(embed_dim,
                                     weight_attr=w_attr_2,
                                     bias_attr=b_attr_2,
                                     epsilon=1e-6)

        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h

        return x


class Encoder(nn.Layer):
    """Transformer encoder
    Encoder encoder contains a list of EncoderLayer, and a LayerNorm.
    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 attn_head_size=None,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        # stochatic depth decay
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         attn_head_size=attn_head_size,
                                         qkv_bias=qkv_bias,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout,
                                         droppath=depth_decay[i])
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights()
        self.encoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out


class VisualTransformer(nn.Layer):
    """ViT transformer
    ViT Transformer, classifier is a single Linear layer for finetune,
    For training from scratch, two layer mlp should be used.
    Classification is done using cls_token.
    Args:
        image_size: int, input image size, default: 224
        patch_size: int, patch size, default: 16
        in_channels: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        embed_dim: int, embedding dimension (patch embed out dim), default: 768
        depth: int, number ot transformer blocks, default: 12
        num_heads: int, number of attention heads, default: 12
        mlp_ratio: float, ratio of mlp hidden dim to embed dim(mlp in dim), default: 4.0
        qkv_bias: bool, If True, enable qkv(nn.Linear) layer with bias, default: True
        dropout: float, dropout rate for linear layers, default: 0.
        attention_dropout: float, dropout rate for attention layers default: 0.
        droppath: float, droppath rate for droppath layers, default: 0.
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 attn_head_size=None,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 train_from_scratch=False):
        super().__init__()
        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              embed_dim,
                                              dropout)
        # create multi head self-attention layers
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               attn_head_size,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath)

        # classifier head (for training from scracth)
        if train_from_scratch:
            w_attr_1, b_attr_1 = self._init_weights()
            w_attr_2, b_attr_2 = self._init_weights()
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim,
                          embed_dim,
                          weight_attr=w_attr_1,
                          bias_attr=b_attr_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim,
                          num_classes,
                          weight_attr=w_attr_2,
                          bias_attr=b_attr_2),
                nn.Dropout(dropout),
            )
        else:
            # classifier head (for finetuning)
            w_attr_1, b_attr_1 = self._init_weights()
            self.classifier = nn.Linear(embed_dim,
                                        num_classes,
                                        weight_attr=w_attr_1,
                                        bias_attr=b_attr_1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        logits = self.classifier(x[:, 0])  # take only cls_token as classifier
        return logits


def build_vit(config):
    """build vit model from config"""
    model = VisualTransformer(image_size=config.DATA.IMAGE_SIZE,
                              patch_size=config.MODEL.TRANS.PATCH_SIZE,
                              in_channels=config.DATA.IMAGE_CHANNELS,
                              num_classes=config.MODEL.NUM_CLASSES,
                              embed_dim=config.MODEL.TRANS.EMBED_DIM,
                              depth=config.MODEL.TRANS.DEPTH,
                              num_heads=config.MODEL.TRANS.NUM_HEADS,
                              attn_head_size=config.MODEL.TRANS.ATTN_HEAD_SIZE,
                              mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                              qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                              dropout=config.MODEL.DROPOUT,
                              attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                              droppath=config.MODEL.DROPPATH,
                              train_from_scratch=False)
    return model
