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
from pos_embed import get_2d_sincos_pos_embed


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_position_encoding(seq_len, embed_dim):
    """ sinusoid position encoding table
    Note: not used in MAE, use get_2d_sincos_pos_embed instead
    """
    def get_position_angle_vec(embed_dim, position):
        return [position / np.power(10000, 2 * (hid_j // 2) / embed_dim) for hid_j in range(embed_dim)]

    sinusoid_table = np.array([get_position_angle_vec(embed_dim, pos_i) for pos_i in range(seq_len)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    position_embedding = paddle.to_tensor([sinusoid_table])
    return position_embedding


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
        self.n_patches = (image_size // patch_size) * (image_size // patch_size)
        w_attr_1, b_attr_1 = self._init_weights()
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         weight_attr=w_attr_1,
                                         bias_attr=b_attr_1)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform()) # MAE 
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = self.dropout(x)
        return x


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
            initializer=nn.initializer.XavierUniform()) # MAE 
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

        q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)
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
        dropout: dropout after fc
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
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform()) # MAE 
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
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
        norm: nn.LayerNorm which is applied after last encoder layer
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 has_norm=True):
        super(Encoder, self).__init__()
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
        self.layers = nn.LayerList(layer_list)
        
        # move this norm out to upper level for global_pool (no cls_token settings)
        self.has_norm = has_norm
        if has_norm:
            w_attr, b_attr = self._init_weights()
            self.norm = nn.LayerNorm(embed_dim,
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

        if self.has_norm:
            x = self.norm(x)
        return x


class Decoder(nn.Layer):
    """Transformer decoder

        Decoder contains a list of TransformerLayer, and a LayerNorm.

        Attributes:
            layers: nn.LayerList contains multiple TransformerLayers
            norm: nn.LayerNorm which is applied after last encoder layer
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
            layer_list.append(TransformerLayer(embed_dim,
                                               num_heads,
                                               qkv_bias,
                                               mlp_ratio,
                                               dropout,
                                               attention_dropout,
                                               droppath=depth_decay[i]))
        self.layers = nn.LayerList(layer_list)

        w_attr, b_attr = self._init_weights()
        self.norm = nn.LayerNorm(embed_dim,
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
        out = self.norm(x)
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
                 decoder_num_heads=16,
                 mlp_ratio=4,
                 qkv_bias=True,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 norm_pix_loss=False):
        super().__init__()
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_size = patch_size
        # -------------------- Encoder --------------------
        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            in_channels,
            encoder_embed_dim,
            dropout)

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, encoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)) #MAE
 
        pos_embed = get_2d_sincos_pos_embed(embed_dim=encoder_embed_dim,
                                            grid_size= int(self.num_patches ** 0.5),
                                            cls_token=True)
        self.encoder_position_embedding = paddle.create_parameter(
            shape=[1, 1 + self.num_patches, encoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(pos_embed, dtype='float32').unsqueeze(0)
            )
        )
        self.encoder_position_embedding.stop_gradient = True

        self.encoder = Encoder(
            encoder_embed_dim,
            encoder_num_heads,
            encoder_depth,
            qkv_bias,
            mlp_ratio,
            dropout,
            attention_dropout,
            droppath)

        # -------------------- Decoder --------------------
        # the embed_dim is different in encoder and decoder, so add a linear layer
        w_attr_1, b_attr_1 = self._init_weights()
        self.linear_projection = nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim,
            weight_attr=w_attr_1,
            bias_attr=b_attr_1)

        self.mask_token = paddle.create_parameter(
            shape=[1, 1, decoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)) #MAE

        pos_embed = get_2d_sincos_pos_embed(embed_dim=decoder_embed_dim,
                                            grid_size= int(self.num_patches ** 0.5),
                                            cls_token=True)
        self.decoder_position_embedding = paddle.create_parameter(
            shape=[1, 1 + self.num_patches, decoder_embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                paddle.to_tensor(pos_embed, dtype='float32').unsqueeze(0)
            )
        )
        self.decoder_position_embedding.stop_gradient = True

        self.decoder = Decoder(
            decoder_embed_dim,
            decoder_num_heads,
            decoder_depth,
            qkv_bias,
            mlp_ratio,
            dropout,
            attention_dropout,
            droppath)

        # create reconstruction layer
        w_attr_2, b_attr_2 = self._init_weights()
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            in_channels * patch_size * patch_size,
            weight_attr=w_attr_2,
            bias_attr=b_attr_2)

        self.norm_pix_loss = norm_pix_loss

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform()) # MAE 
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def patchify(self, images):
        n_patches = images.shape[2] // self.patch_size
        x = images.reshape([images.shape[0], # N
                            images.shape[1], # C
                            n_patches, # h
                            self.patch_size, # p
                            n_patches, # w
                            self.patch_size]) # p
        x = x.transpose([0, 2, 4, 3, 5, 1])
        x = x.reshape([images.shape[0], n_patches * n_patches, -1])
        return x

    def unpatchify(self, x):
        n_patches = int(x.shape[1]**.5) 

        x = x.reshape([x.shape[0], # N
                       n_patches, # h
                       n_patches, # w
                       self.patch_size, # p
                       self.patch_size, # p
                       -1]) # C
        x = x.transpose([0, 5, 1, 3, 2, 4])
        x = x.reshape([images.shape[0], -1, n_patches * self.patch_size, n_patches * self.patch_size])
        return x

    def random_masking(self, x, mask_ratio, rand_probs=None):
        """
        Shuffle x then mask the last few tokens according to mask ratio.
        Args:
            x: tensor of [batch, seq_len, encoder_embed_dim]
            mask_ratio: float, masking ratio
        Returns:
            masked_x: tensor of [batch, seq_len - mask_num, encoder_embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        keep_len = int(seq_len * (1 - mask_ratio))
        # for debug only
        rand_probs = rand_probs if rand_probs is not None else paddle.rand([batch_size, seq_len])
        #rand_probs = paddle.rand([batch_size, seq_len])
        shuffle_ids = paddle.argsort(rand_probs, axis=-1)
        restore_ids = paddle.argsort(shuffle_ids, axis=-1)

        keep_ids = shuffle_ids[:, :keep_len]
        ids = keep_ids + (paddle.arange(batch_size) * seq_len).unsqueeze(-1).expand([batch_size, -1])
        x_masked = paddle.gather(x.flatten(0, 1), index=ids.flatten(), axis=0).reshape([batch_size, keep_len, -1])

        mask = paddle.ones([batch_size, seq_len])
        mask[:, :keep_len] = 0

        restore_ids_expand = restore_ids + (paddle.arange(batch_size) * seq_len).unsqueeze(-1).expand([batch_size, -1])
        mask = paddle.gather(mask.flatten(), index=restore_ids_expand.flatten()).reshape([batch_size, seq_len])
        return x_masked, mask, restore_ids

    def forward_encoder(self, images, mask_ratio, rand_probs=None):
        x = self.patch_embedding(images)
        # add pos embed w/o cls token
        x = x + self.encoder_position_embedding[:, 1:, :]
        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio, rand_probs)
        # append cls token
        cls_token = self.cls_token + self.encoder_position_embedding[:, :1, :]
        cls_tokens = cls_token.expand((x.shape[0], -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x  = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.linear_projection(x) # [batch, keep_len+1(cls_token), decoder_embed_dim]
        # self.mask_token: [1, 1, decoder_embed_dim]
        # ids_store: [batch, num_patches]
        # mask_tokens: [batch, masked_len, decoder_embed_dim]
        mask_tokens = self.mask_token.expand([x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1])
        # x_: [batch, num_patches, decoder_embed_dim] 
        x_ = paddle.concat([x[:, 1:, :], mask_tokens], axis=1) # no cls token 
        x_shape = x_.shape
        batch_size = x_shape[0]
        seq_len = x_shape[1]

        ## The following ops assures the paddle gather_nd op has the same behaviour as pytorch gather op.
        ids_restore_expand = ids_restore + (paddle.arange(batch_size) * seq_len).unsqueeze(-1).expand([batch_size, -1])
        x_ = paddle.gather_nd(x_.flatten(0, 1), index=ids_restore_expand.flatten().unsqueeze(-1)) 
        x_ = x_.reshape(x_shape)

        x = paddle.concat([x[:, :1, :], x_], axis=1) # append cls token

        x = x + self.decoder_position_embedding
        x = self.decoder(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, images, pred, mask):
        target = self.patchify(images)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keepdim=True)
            var = target.var(axis=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1) # mean loss per patch
        loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
        return loss

    def forward(self, images, mask_ratio=0.75, rand_probs=None):
        encoder_out, mask, restore_ids = self.forward_encoder(images, mask_ratio, rand_probs)
        decoder_out = self.forward_decoder(encoder_out, restore_ids)
        loss = self.forward_loss(images, decoder_out, mask)
        return loss, decoder_out, mask


class MAETransformer(nn.Layer):
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
                 global_pool=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        self.global_pool = global_pool
        # create patch embedding with positional embedding
        self.patch_embedding = PatchEmbedding(image_size,
                                              patch_size,
                                              in_channels,
                                              embed_dim,
                                              dropout)
        # create positional embedding
        self.encoder_position_embedding = paddle.create_parameter(
            shape=[1, 1 + self.patch_embedding.n_patches, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(
                get_position_encoding(seq_len=1 + self.patch_embedding.n_patches,
                                      embed_dim=embed_dim)
            )
        )
        # create class token
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0))
        # create multi head self-attention encoder
        self.encoder = Encoder(embed_dim,
                               num_heads,
                               depth,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout,
                               droppath,
                               has_norm=not global_pool)
        # define encoder norm here to aviod cls_token (when global_pool is True)
        if global_pool:
            w_attr, b_attr = self._init_weights_norm()
            self.encoder_norm = nn.LayerNorm(embed_dim,
                                             weight_attr=w_attr,
                                             bias_attr=b_attr,
                                             epsilon=1e-6)

        # classifier head (for finetuning)
        w_attr_1, b_attr_1 = self._init_weights_classifier()
        self.classifier = nn.Linear(embed_dim,
                                    num_classes,
                                    weight_attr=w_attr_1,
                                    bias_attr=b_attr_1)


    def forward_features(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.encoder_position_embedding
        x = self.encoder(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(axis=1) # global pool w/o cls_token
            out = self.encoder_norm(x)
        else:
            # norm is applied in encoder
            out = x[:, 0] # return cls_token only

        return out

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.classifier(x)

        return logits

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02)) # MAE linearprobe 
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def _init_weights_classifier(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.01))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        return weight_attr, bias_attr


def build_mae_pretrain(config):
    """ build MAE vit model for pretraining"""
    model = MAEPretrainTransformer(image_size=config.DATA.IMAGE_SIZE,
                                   patch_size=config.MODEL.PATCH_SIZE,
                                   in_channels=config.DATA.IMAGE_CHANNELS,
                                   encoder_embed_dim=config.MODEL.ENCODER.EMBED_DIM,
                                   decoder_embed_dim=config.MODEL.DECODER.EMBED_DIM,
                                   encoder_depth=config.MODEL.ENCODER.DEPTH,
                                   decoder_depth=config.MODEL.DECODER.DEPTH,
                                   encoder_num_heads=config.MODEL.ENCODER.NUM_HEADS,
                                   decoder_num_heads=config.MODEL.DECODER.NUM_HEADS,
                                   mlp_ratio=config.MODEL.MLP_RATIO,
                                   qkv_bias=config.MODEL.QKV_BIAS,
                                   dropout=config.MODEL.DROPOUT,
                                   attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                                   droppath=config.MODEL.DROPPATH,
                                   norm_pix_loss=config.MODEL.NORM_PIX_LOSS)
    return model


def build_transformer(config):
    """ build vit model for finetuning and linear probing"""
    model = MAETransformer(image_size=config.DATA.IMAGE_SIZE,
                           patch_size=config.MODEL.PATCH_SIZE,
                           in_channels=config.DATA.IMAGE_CHANNELS,
                           num_classes=config.MODEL.NUM_CLASSES,
                           embed_dim=config.MODEL.ENCODER.EMBED_DIM,
                           depth=config.MODEL.ENCODER.DEPTH,
                           num_heads=config.MODEL.ENCODER.NUM_HEADS,
                           mlp_ratio=config.MODEL.MLP_RATIO,
                           qkv_bias=config.MODEL.QKV_BIAS,
                           global_pool=config.MODEL.GLOBAL_POOL,
                           dropout=config.MODEL.DROPOUT,
                           attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                           droppath=config.MODEL.DROPPATH)
    return model
