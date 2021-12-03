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


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PositionalEmbedding(nn.Layer):
    """Position Embedding

    Apply positional embedding on input images.

    Attributes:
        encoder_position_embedding: sine-cosine version positional embedding
        decoder_position_embedding: sine-cosine version positional embedding
    """

    def __init__(self, encoder_embed_dim, decoder_embed_dim, seq_len=197):
        """ Sinusoid position encoding table """
        super(PositionalEmbedding, self).__init__()
        self.seq_len = seq_len

        def get_position_angle_vec(embed_dim, position):
            return [position / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

        sinusoid_table = np.array([get_position_angle_vec(
            encoder_embed_dim, pos_i) for pos_i in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        encoder_position_embedding = paddle.to_tensor([sinusoid_table])

        sinusoid_table = np.array([get_position_angle_vec(
            decoder_embed_dim, pos_i) for pos_i in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        decoder_position_embedding = paddle.to_tensor([sinusoid_table])

        self.register_buffer('encoder_position_embedding',
                             encoder_position_embedding)
        self.register_buffer('decoder_position_embedding',
                             decoder_position_embedding)

    def get_encoder_embedding(self, seq_length=None):
        if seq_length is None:
            seq_length = self.seq_len
        return self.encoder_position_embedding[:, :seq_length, :]

    def get_decoder_embedding(self, seq_length=None):
        if seq_length is None:
            seq_length = self.seq_len
        return self.decoder_position_embedding[:, :seq_length, :]


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
            initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingUniform())
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
        attn_weights = attn
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z.transpose([0, 2, 1, 3])
        new_shape = z.shape[:-2] + [self.all_head_size]
        z = z.reshape(new_shape)
        # reshape
        z = self.out(z)
        z = self.proj_dropout(z)
        return z, attn_weights


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
            initializer=paddle.nn.initializer.XavierUniform())  # default in pp: xavier
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=1e-6))  # default in pp: zero
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
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        return weight_attr, bias_attr

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x, attn = self.attn(x)
        x = self.drop_path(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h

        return x, attn


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
        super(Encoder, self).__init__()
        # stochatic depth decay
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]
        layer_list = []
        for i in range(depth):
            encoder_layer = TransformerLayer(embed_dim,
                                             num_heads,
                                             qkv_bias,
                                             mlp_ratio,
                                             dropout,
                                             attention_dropout,
                                             droppath=depth_decay[i])
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights()
        self.encoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        return weight_attr, bias_attr

    def forward(self, x):
        self_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            self_attn.append(attn)
        out = self.encoder_norm(x)
        return out, self_attn


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
        super(Decoder, self).__init__()
        # stochatic depth decay
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            decoder_layer = TransformerLayer(embed_dim,
                                             num_heads,
                                             qkv_bias,
                                             mlp_ratio,
                                             dropout,
                                             attention_dropout,
                                             droppath=depth_decay[i])
            layer_list.append(copy.deepcopy(decoder_layer))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = self._init_weights()
        self.decoder_norm = nn.LayerNorm(embed_dim,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1,
                                         epsilon=1e-6)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        return weight_attr, bias_attr

    def forward(self, x):
        self_attn = []
        for layer in self.layers:
            x, attn = layer(x)
            self_attn.append(attn)
        out = self.decoder_norm(x)
        return out, self_attn


class MAEPretrainTransformer(nn.Layer):
    # TODO: 补注释
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
                 mask_ratio=0.75,
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
                 droppath=0.,
                 train_from_scratch=False,
                 config=None):
        super(MAEPretrainTransformer, self).__init__()
        self.patch_size = patch_size
        # mask-related parameters
        self.mask_ratio = mask_ratio
        self.mask_token = paddle.create_parameter(
            shape=[1, 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0))
        self.perm = None
        self.mask_num = None
        # create positional embedding
        self.position_embedding = PositionalEmbedding(encoder_embed_dim, decoder_embed_dim,
                                                      int(1 + (image_size / patch_size) * (image_size / patch_size)))
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
        self.linear_projection = nn.Linear(encoder_embed_dim,
                                           decoder_embed_dim)
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
        self.reconstruction_layer = nn.Linear(decoder_embed_dim,
                                              in_channels * patch_size * patch_size)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, image):
        source = self.patch_embedding(image)
        target = image

        # mask source before encoding
        source = self.mask(source)
        # add positional embedding before encoding
        source += self.position_embedding.get_encoder_embedding(source.shape[1])
        source, attn = self.encoder(source)
        source = self.linear_projection(source)
        source = self.unmask(
            source, self.position_embedding.get_decoder_embedding())
        source, attn = self.decoder(source)

        # only sustain unmasked target patches
        masked_target = self.mask_target(target, self.patch_size)
        # only reconstruct unmasked source patches
        _, unmask_length, _ = masked_target.shape
        reconstructed_image = self.reconstruction_layer(
            source[:, 1: unmask_length + 1, :])
        return reconstructed_image, masked_target

    def mask(self, x):
        """
        Shuffle x then mask the last few tokens according to mask ratio.
        Args:
            x: tensor of [batch, seq_len + 1, encoder_embed_dim]
            note the extra 1 is corresponding to [cls] token

        Returns:
            masked_x: tensor of [batch, seq_len + 1 - mask_num, encoder_embed_dim]
        """
        _, seq_len, _ = x.shape
        seq_len = seq_len - 1  # should not shuffle the [cls] token
        self.mask_num = int(seq_len * self.mask_ratio)
        # should not shuffle the [cls] token
        index = [i for i in range(1, seq_len + 1)]
        random.shuffle(index)
        self.perm = paddle.to_tensor([0] + index)  # add back [cls] token
        shuffled_x = paddle.index_select(x, self.perm, axis=1)
        masked_x = shuffled_x[:, 0: -self.mask_num, :]
        return masked_x

    def mask_target(self, target, patch_size):
        """
        Shuffle and mask label
        According to the paper, the reconstruction loss is only calculated
        over unmasked tokens.
        Args:
            patch_size: int
            target: tensor of [batch, channel, image_size, image_size]

        Returns:
            masked_label: tensor of [batch, seq_len-mask_num, channel * patch_size * patch_size]
            note that seq_len = (image_size / patch_size) ^ 2
        """
        shuffled_target = F.unfold(
            target, patch_size, patch_size).transpose((0, 2, 1))
        # shuffled_label shape is [batch, seq_len , channel * patch_size * patch_size]

        masked_target = shuffled_target[:, 0: -self.mask_num, :]
        return masked_target

    def unmask(self, x, pos_embedding):
        """
        Add back mask tokens. Then add shuffled positional embedding
        Args:
            x: tensor of [batch, seq_len + 1 - mask_num, decoder_embed_dim]
            pos_embedding: tensor of [batch, seq_len + 1, decoder_embed_dim]

        Returns:
            unmasked_x: tensor of [batch, seq_len + 1, decoder_embed_dim]
        """
        batch, _, _ = x.shape
        mask_tokens = self.mask_token.expand((batch, self.mask_num, -1))
        # [batch, seq_len + 1, decoder_embed_dim]
        unmasked_x = paddle.concat([x, mask_tokens], axis=1)
        shuffled_pos_embedding = paddle.index_select(
            pos_embedding, self.perm, axis=1)
        return unmasked_x + shuffled_pos_embedding

class MAEFinetuneTransformer(nn.Layer):
    # TODO: 补注释
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
                 num_classes=1000,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 train_from_scratch=False,
                 config=None):
        super(MAEFinetuneTransformer, self).__init__()
        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              encoder_embed_dim,
                                              dropout)
        # create multi head self-attention encoder
        self.encoder = Encoder(encoder_embed_dim,
                               num_heads,
                               encoder_depth,
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
                nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                          config.MODEL.TRANS.HIDDEN_SIZE,
                          weight_attr=w_attr_1,
                          bias_attr=b_attr_1),
                nn.ReLU(),
                nn.Dropout(config.MODEL.DROPOUT),
                nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                          config.MODEL.NUM_CLASSES,
                          weight_attr=w_attr_2,
                          bias_attr=b_attr_2),
                nn.Dropout(config.MODEL.DROPOUT),
            )
        else:
            # classifier head (for finetuning)
            w_attr_1, b_attr_1 = self._init_weights()
            self.classifier = nn.Linear(encoder_embed_dim,
                                        num_classes,
                                        weight_attr=w_attr_1,
                                        bias_attr=b_attr_1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.patch_embedding(x)
        x, attn = self.encoder(x)
        logits = self.decoder(x)
        return logits


def build_mae_pretrain(config):
    model = MAEPretrainTransformer(image_size=config.DATA.IMAGE_SIZE,
                                   patch_size=config.MODEL.TRANS.PATCH_SIZE,
                                   in_channels=3,
                                   mask_ratio=config.MODEL.TRANS.MASK_RATIO,
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
                                   droppath=config.MODEL.DROPPATH,
                                   train_from_scratch=False,
                                   config=config)
    return model


def build_mas_finetune(config):
    # TODO: to be implemented
    pass
