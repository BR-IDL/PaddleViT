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
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from droppath import DropPath
from config import get_config


def get_position_encoding(seq_len, embed_dim):
    """ sinusoid position encoding table"""
    def get_position_angle_vec(embed_dim, position):
        return [position / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_position_angle_vec(embed_dim, pos_i) for pos_i in range(seq_len)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    position_embedding = paddle.to_tensor([sinusoid_table])
    return position_embedding


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PositionalEmbedding(nn.Layer):
    """Position Embedding

    Apply positional embedding on input images.

    Attributes:
        position_embedding: sine-cosine version positional embedding
    """
    def __init__(self, embed_dim, seq_len=197):
        """ Sinusoid position encoding table """
        super().__init__()
        self.seq_len = seq_len

        def get_position_angle_vec(embed_dim, position):
            return [position / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

        sinusoid_table = np.array([get_position_angle_vec(
            embed_dim, pos_i) for pos_i in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        position_embedding = paddle.to_tensor([sinusoid_table])

        self.register_buffer('position_embedding',
                             position_embedding)

    def get_positional_embedding(self, seq_length=None):
        if seq_length is None:
            seq_length = self.seq_len
        return self.position_embedding[:, :seq_length, :]


class PatchEmbedding(nn.Layer):
    """Patch Embedding

    Apply patch embedding on input images.

    Attributes:
        patch_embddings: impl using a patch_size x patch_size Conv2D operation
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

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_tokens = self.cls_token.expand(
            (x.shape[0], -1, -1))
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat((cls_tokens, x), axis=1)
        embeddings = self.dropout(x)
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
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_size = int(embed_dim / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_size * 3,  # weights for q, k, and v
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_size ** -0.5

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(embed_dim,
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
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
            initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
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


class TransformerLayer(nn.Layer):
    """Transformer Layer

    Transformer Layer contains attention, norm, mlp and residual

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
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
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

    Encoder contains a list of TransformerLayer, and a LayerNorm.

    Attributes:
        layers: nn.LayerList contains multiple TransformerLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
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
            layer_list.append(TransformerLayer(embed_dim,
                                             num_heads,
                                             qkv_bias,
                                             mlp_ratio,
                                             dropout,
                                             attention_dropout,
                                             droppath=depth_decay[i]))
            # new paddle version fix this, deepcopy is no longer needed
            # layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.LayerList(layer_list)
        
        w_attr, b_attr = self._init_weights()
        self.encoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr,
                                         bias_attr=b_attr,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out


class Decoder(nn.Layer):
    """Transformer decoder

        Decoder contains a list of TransformerLayer, and a LayerNorm.

        Attributes:
            layers: nn.LayerList contains multiple TransformerLayers
            decoder_norm: nn.LayerNorm which is applied after last encoder layer
        """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
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
            layer_list.append(TransformerLayer(embed_dim,
                                               num_heads,
                                               qkv_bias,
                                               mlp_ratio,
                                               dropout,
                                               attention_dropout,
                                               droppath=depth_decay[i]))
            # new paddle version fix this, deepcopy is no longer needed
            # layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.LayerList(layer_list)

        w_attr, b_attr = self._init_weights()
        self.decoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr,
                                         bias_attr=b_attr,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, mask_len=0):
        for layer in self.layers:
            x = layer(x)
        if mask_len > 0:
            # only sustain masked patches
            out = self.decoder_norm(x[:, -mask_len:])
        else:
            out = self.decoder_norm(x)
        return out


class MAEPretrainTransformer(nn.Layer):
    """ViT transformer

    ViT Transformer, classifier is a single Linear layer for finetune,
    For training from scratch, two layer mlp should be used.
    Classification is done using cls_token.

    Args:
        image_size: int, input image size, default: 224
        patch_size: int, patch size, default: 16
        in_channels: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        encoder_embed_dim: int, embedding dimension (patch embed out dim), default: 768
        decoder_embed_dim: int, embedding dimension (patch embed out dim), default: 512
        encoder_depth: int, number ot transformer blocks, default: 12
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
                 encoder_embed_dim=768,
                 decoder_embed_dim=512,
                 encoder_depth=12,
                 decoder_depth=8,
                 encoder_num_heads=12,
                 decoder_num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.mask_token = paddle.create_parameter(
            shape=[1, 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        self.perm = None
        self.mask_num = None
        # create positional embedding
        self.encoder_position_embedding = get_position_encoding(seq_len=1 + self.num_patches,
                                                                embed_dim=encoder_embed_dim) 
        self.decoder_position_embedding = get_position_encoding(seq_len=1 + self.num_patches,
                                                                embed_dim=decoder_embed_dim) 
        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              encoder_embed_dim,
                                              dropout)
        # create multi head self-attention encoder
        self.encoder = Encoder(encoder_embed_dim,
                               encoder_num_heads,
                               encoder_depth,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath)
        # the embed_dim is different in encoder and decoder, so add a linear layer
        w_attr_1, b_attr_1 = self._init_weights()
        self.linear_projection = nn.Linear(encoder_embed_dim,
                                           decoder_embed_dim,
                                           weight_attr=w_attr_1,
                                           bias_attr=b_attr_1)
        # create multi head self-attention decoder
        self.decoder = Decoder(decoder_embed_dim,
                               decoder_num_heads,
                               decoder_depth,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath)
        # create reconstruction layer
        w_attr_2, b_attr_2 = self._init_weights()
        self.reconstruction_layer = nn.Linear(decoder_embed_dim,
                                              in_channels * patch_size * patch_size,
                                              weight_attr=w_attr_2,
                                              bias_attr=b_attr_2)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x, masks):
        # x: [B, C, H, W]
        x = self.patch_embedding(x)
        # x: [B, num_patches, embed_dim]
        B, N, C = x.shape # B: batch_size, N: num_patches, C: embed_dim
        # mask: [B, num_patches], visible set to 0, masked set to 1

        # add pos embed
        x += self.encoder_position_embedding.clone().detach()
        # get no mask patches
        no_mask_x = x[~masks] # [B*0.25*L, embed_dim]
        # index slicing needs reshape back in paddle: [B, 0.25L, embed_dim]
        no_mask_x = no_mask_x.reshape([B, -1, C])
        # encoder
        enc_out = self.encoder(no_mask_x)
        # encoder to decoder linear proj
        enc_out = self.linear_projection(enc_out)
        # shuffle the position embedding is equivalent to unshuffling tokens 
        expand_pos_embed = self.decoder_position_embedding.expand([B, -1, -1]).clone().detach()
        pos_embed_no_mask = expand_pos_embed[~masks].reshape([B, -1, enc_out.shape[-1]])
        pos_embed_mask = expand_pos_embed[masks].reshape([B, -1, enc_out.shape[-1]])
        # dec in put, here use broadcasting for mask_token
        dec_in = paddle.concat([enc_out + pos_embed_no_mask, self.mask_token + pos_embed_mask], axis=1)
        # decoder
        mask_len = pos_embed_mask.shape[1]
        dec_out = self.decoder(dec_in, mask_len)
        # reconstruct patches
        output = self.reconstruction_layer(dec_out)
        return output


class MAEFinetuneTransformer(nn.Layer):
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
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        # create positional embedding
        self.encoder_position_embedding = get_position_encoding(seq_len=1 + self.num_patches,
                                                                embed_dim=embed_dim) 
        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              embed_dim,
                                              dropout)
        # create multi head self-attention encoder
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath)

        # classifier head (for finetuning)
        w_attr_1, b_attr_1 = self._init_weights()
        self.classifier = nn.Linear(embed_dim,
                                    num_classes,
                                    weight_attr=w_attr_1,
                                    bias_attr=b_attr_1)

    def forward(self, x):
        x = self.patch_embedding(x)
        # add pos embed
        x += self.encoder_position_embedding.clone().detach()
        x = self.encoder(x)
        logits = self.classifier(x[:, 0])  # take only cls_token as classifier
        return logits

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr


def build_mae_pretrain(config):
    model = MAEPretrainTransformer(image_size=config.DATA.IMAGE_SIZE,
                                   patch_size=config.MODEL.TRANS.PATCH_SIZE,
                                   in_channels=3,
                                   encoder_embed_dim=config.MODEL.TRANS.ENCODER.EMBED_DIM,
                                   decoder_embed_dim=config.MODEL.TRANS.DECODER.EMBED_DIM,
                                   encoder_depth=config.MODEL.TRANS.ENCODER.DEPTH,
                                   decoder_depth=config.MODEL.TRANS.DECODER.DEPTH,
                                   encoder_num_heads=config.MODEL.TRANS.ENCODER.NUM_HEADS,
                                   decoder_num_heads=config.MODEL.TRANS.DECODER.NUM_HEADS,
                                   mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                                   qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                                   dropout=config.MODEL.DROPOUT,
                                   attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                                   droppath=config.MODEL.DROPPATH)
    return model


def build_mae_finetune(config):
    model = MAEFinetuneTransformer(image_size=config.DATA.IMAGE_SIZE,
                                   patch_size=config.MODEL.TRANS.PATCH_SIZE,
                                   in_channels=3,
                                   embed_dim=config.MODEL.TRANS.ENCODER.EMBED_DIM,
                                   depth=config.MODEL.TRANS.ENCODER.DEPTH,
                                   num_heads=config.MODEL.TRANS.ENCODER.NUM_HEADS,
                                   mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                                   qkv_bias=config.MODEL.TRANS.QKV_BIAS,
                                   dropout=config.MODEL.DROPOUT,
                                   attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                                   droppath=config.MODEL.DROPPATH)
    return model
