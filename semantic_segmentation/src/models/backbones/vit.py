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

"""
Implement Transformer Class for ViT
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from src.utils import load_pretrained_model


class Embeddings(nn.Layer):
    """Patch Embeddings and Position Embeddings

    Apply patch embeddings and position embeddings on input images.
    Currently hybrid is not supported yet.

    Attributes:
        hybrid: TODO.
        patch_embddings: impl using a patch_size x patch_size Conv2D operation
        position_embddings: a parameter with len = num_patch + 1(for cls_token)
        cls_token: token insert to the patch feature for classification
        dropout: dropout for embeddings
    """

    def __init__(self, config, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = config.MODEL.TRANS.HYBRID
        image_size = config.DATA.CROP_SIZE
        self.keep_cls_token = config.MODEL.TRANS.KEEP_CLS_TOKEN
        if self.hybrid:
            #TODO: add resnet model 
            self.hybrid_model = None

        if config.MODEL.TRANS.PATCH_GRID is not None:
            self.hybrid = True
            grid_size = config.MODEL.TRANS.PATCH_GRID
            patch_size = (image_size[0] // 16 // grid_size, image_size[1] // 16 // grid_size)
            n_patches = (image_size[0] // 16) * (image_size[1] // 16)
        else:
            self.hybrid = False
            patch_size = config.MODEL.TRANS.PATCH_SIZE
            n_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        self.patch_embeddings = nn.Conv2D(in_channels=in_channels,
                                          out_channels=config.MODEL.TRANS.HIDDEN_SIZE,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.position_embeddings = paddle.create_parameter(
                                    shape=[1, n_patches+1, config.MODEL.TRANS.HIDDEN_SIZE],
                                    dtype='float32',
                                    default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)) # may be important

        self.cls_token = paddle.create_parameter(
                                    shape=[1, 1, config.MODEL.TRANS.HIDDEN_SIZE],
                                    dtype='float32',
                                    default_initializer=paddle.nn.initializer.Constant(0))

        self.dropout = nn.Dropout(config.MODEL.DROPOUT)

    def forward(self, x):
        cls_tokens = self.cls_token[0].expand((x.shape[0], -1, -1))
        if self.hybrid:
            # x = self.hybrid_model(x)  # TODO
            pass
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat((cls_tokens, x), axis=1)
        embeddings = x + self.position_embeddings[0] # tensor broadcast
        if not self.keep_cls_token:
            embeddings = embeddings[:, 1:]  # For SETR 
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

    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_heads = config.MODEL.TRANS.NUM_HEADS
        self.attn_head_size = int(config.MODEL.TRANS.HIDDEN_SIZE / self.num_heads)
        self.all_head_size = self.attn_head_size * self.num_heads

        w_attr_1, b_attr_1 = self._init_weights()
        self.qkv = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             self.all_head_size*3,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1 if config.MODEL.TRANS.QKV_BIAS else False)

        self.scales = self.attn_head_size ** -0.5  # 0.125 for Large

        w_attr_2, b_attr_2 = self._init_weights()
        self.out = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             config.MODEL.TRANS.HIDDEN_SIZE,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(config.MODEL.ATTENTION_DROPOUT)
        self.proj_dropout = nn.Dropout(config.MODEL.DROPOUT)

        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
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

    def __init__(self, config):
        super(Mlp, self).__init__()

        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(config.MODEL.TRANS.HIDDEN_SIZE,
                             int(config.MODEL.TRANS.MLP_RATIO * config.MODEL.TRANS.HIDDEN_SIZE),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(int(config.MODEL.TRANS.MLP_RATIO * config.MODEL.TRANS.HIDDEN_SIZE),
                             config.MODEL.TRANS.HIDDEN_SIZE,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU() 
        self.dropout1 = nn.Dropout(config.MODEL.DROPOUT)
        #self.dropout2 = nn.Dropout(config.MODEL.DROPOUT)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()) #default in pp: xavier
        bias_attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=1e-6)) #default in pp: zero
        
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout1(x)
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

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.hidden_size = config.MODEL.TRANS.HIDDEN_SIZE
        self.attn_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.mlp_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.mlp = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x, attn = self.attn(x)
        x = x + h 

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x, attn


class Encoder(nn.Layer):
    """Encoder

    Encoder contains a list of EncoderLayer, and a LayerNorm at the end.

    Attributes:
        layers: nn.LayerList contains multiple EncoderLayers
        encoder_norm: nn.LayerNorm which is applied after last encoder layer
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList([EncoderLayer(config) for _ in range(config.MODEL.TRANS.NUM_LAYERS)])
        #self.encoder_norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE, epsilon=1e-6)
        self.out_idx_list = tuple(range(config.MODEL.TRANS.NUM_LAYERS))

    def forward(self, x):
        self_attn = []
        outs = []
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x)
            self_attn.append(attn)
            if layer_idx in self.out_idx_list:
                outs.append(x)
        #out = self.encoder_norm(x)
        return outs


class Transformer(nn.Layer):
    """Transformer

    Attributes:
        embeddings: patch embeddings and position embeddings
        encoder: encoder layers with multihead self attention
    """

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.dropout = nn.Dropout(config.MODEL.DROPOUT)
        self.encoder = Encoder(config)

    def forward(self, x):
        embedding_out = self.embeddings(x)
        embedding_out = self.dropout(embedding_out)
        encoder_outs = self.encoder(embedding_out)
        return encoder_outs

class VisualTransformer(nn.Layer):
    """ VisualTransformer
   
    Vision Transformer as the backbone of SETR-PUP and SETR-Naive. 
    Ref. https://arxiv.org/pdf/2012.15840.pdf

    """
    def __init__(self, config):
        super(VisualTransformer, self).__init__()
        self.transformer = Transformer(config)
        self.out_indices = config.MODEL.ENCODER.OUT_INDICES
        norm_weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        if config.MODEL.PRETRAINED is not None:
            load_pretrained_model(self, config.MODEL.PRETRAINED)

    def forward(self, x):
        B = x.shape[0]
        outs = self.transformer(x)
        feats = [ ]
        # getting multi-level feats from layers
        for idx in self.out_indices:
            feats.append(outs[idx])
        return feats 

