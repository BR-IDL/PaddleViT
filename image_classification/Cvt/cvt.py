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

import paddle
import paddle.nn as nn

from numpy import repeat
import os
from drop import DropPath


def graph2vector(x: paddle.Tensor):
    '''
    handle the tensor's dimension, expanding images into tensors.
    b c h w -> b (h w) c.
    b c h w mean the quantity of picures is b,number of channels is c.
    each picture size is h*w.
    '''
    B, C, H, W = x.shape
    x = paddle.transpose(x, [0, 2, 3, 1])
    x = paddle.reshape(x, [B, H*W, C])
    return x


def vector2graph(x: paddle.Tensor, h, w):
    '''
    handle the tensor's dimension, take tensor into images.
    b (h w) c -> b c h w.
    b c h w mean the quantity of picures is b,number of channels is c
    each picture size is h*w.

    '''
    B, L, C = x.shape  # L is length of tensor
    x = paddle.transpose(x, [0, 2, 1])
    x = paddle.reshape(x, [B, C, h, w])
    return x


def multitoken(x, h):
    '''
    expand the dim of x.
    handle the tensor's dimension, get multi-head token.
    b t (h d) -> b h t d
    '''
    B, T, L = x.shape
    x = paddle.reshape(x, [B, T, h, -1])
    x = paddle.transpose(x, [0, 2, 1, 3])
    return x


class RearrangeLayer(nn.Layer):
    '''rearrange layer module
    b c h w -> b (h w) c
    a layer to expand x:imgaes into tensors.
    b c h w mean the quantity of picures is b,number of channels is c
    each picture size is h*w.
    '''

    def forward(self, x: paddle.Tensor):
        return graph2vector(x)


class QuickGELU(nn.Layer):
    '''
    Rewrite GELU function to increase processing speed
    '''

    def forward(self, x: paddle.Tensor):
        return x * nn.functional.sigmoid(1.702 * x)


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
                 act_layer=nn.GELU,
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
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(std=1e-6))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding
    using nn.Conv2D and norm_layer to embedd the input.
    Ops: conv -> norm.
    Attributes:
        conv: nn.Conv2D
        norm: nn.LayerNorm
    nn.LayerNorm handle thr input with one dim, so we should
    stretch 2D input into 1D

    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        # conv patch_size to a square,which shape is(patch_size,patch_size)
        patch_size = tuple(repeat((patch_size), 2))

        self.patch_size = patch_size
        self.proj = nn.Conv2D(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = graph2vector(x)
        if self.norm:
            x = self.norm(x)
        x = vector2graph(x, H, W)
        return x


class Attention(nn.Layer):
    """ Attention module
    Attention module for CvT.
    using conv to calculate q,k,v
    Attributes:
        num_heads: number of heads
        qkv: a nn.Linear for q, k, v mapping
            dw_bn: nn.Conv2D -> nn.BatchNorm
            avg: nn.AvgPool2D
            linear: None
        scales: 1 / sqrt(single_head_feature_dim)
        attn_drop: dropout for attention
        proj_drop: final dropout before output
        out: projection of multi-head attention
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 kernel_size=3,
                 stride_kv=2,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        # init to save the pararm
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        # calculate q,k,v with conv
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, 
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv,
        )

        # init parameters of q,k,v
        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        # init project other parameters
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                         ):
        
        proj = nn.Sequential(
            (nn.Conv2D(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias_attr=False,
                groups=dim_in
            )),
            (nn.BatchNorm2D(dim_in)),
            (RearrangeLayer()),
        )

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:  # spilt token from x
            cls_token, x = paddle.split(x, [1, h*w], 1)
        x = vector2graph(x, h, w)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = graph2vector(x)
        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = graph2vector(x)
        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = graph2vector(x)
        if self.with_cls_token:
            q = paddle.concat([cls_token, q], axis=1)
            k = paddle.concat([cls_token, k], axis=1)
            v = paddle.concat([cls_token, v], axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):  # if not generate q,k,v with Linear param
            q, k, v = self.forward_conv(x, h, w)

        # now q,k,v is b (h w) c

        q = multitoken(self.proj_q(
            q), h=self.num_heads)
        k = multitoken(self.proj_k(
            k),  h=self.num_heads)
        v = multitoken(self.proj_v(
            v),  h=self.num_heads)


        # multi tensor with axis=3，then * scale，achieve the result of q*k/sqort(d_k),
        attn_score = paddle.matmul(q, k, transpose_y=True) * self.scale
        attn = nn.functional.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v)
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [0, 0, -1])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # b,t,(h,d)


class Block(nn.Layer):
    ''' Block moudule
    Ops: token -> multihead attention (reshape token to a grap) ->Mlp->token
    '''

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            dim_out,
            mlp_ratio,
            act_layer=act_layer,
            dropout=drop
        )

    def forward(self, x, h, w):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Layer):
    """ VisionTransformer moudule
    Vision Transformer with support for patch or hybrid CNN input stage
    Ops:intput -> conv_embed -> depth*block -> out
    Attribute:
        input: raw picture
        out: features,cls_token

    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=QuickGELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']

        if with_cls_token:
            self.cls_token = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype='float32',
                default_initializer=nn.initializer.TruncatedNormal(std=.02))
            #self.cls_token = paddle.zeros([1, 1, embed_dim])
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.LayerList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            #logging.info('=> init weight of Linear from trunc norm')
            trun_init = nn.initializer.TruncatedNormal(std=0.02)
            trun_init(m.weight)
            if m.bias is not None:
                #logging.info('=> init bias of Linear to zeros')
                zeros = nn.initializer.Constant(0.)
                zeros(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros = nn.initializer.Constant(0.)
            zeros(m.bias)
            ones = nn.initializer.Constant(1.0)
            ones(m.weight)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            #logging.info('=> init weight of Linear from xavier uniform')
            xavier_init = nn.initializer.XavierNormal()
            xavier_init(m.weight)
            if m.bias is not None:
                #logging.info('=> init bias of Linear to zeros')
                zeros = nn.initializer.Constant(0.)
            zeros(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros = nn.initializer.Constant(0.)
            zeros(m.bias)
            ones = nn.initializer.Constant(1)
            ones(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = graph2vector(x)
        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = paddle.expand(self.cls_token, [B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H*W], 1)
        x = vector2graph(x,  H, W)
        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Layer):
    '''Cvt moudule
    using Convolutional Neural Network in attention moudule and embedding
    Args:

        in_chans: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        num_stage: int, numebr of stage, length of array of parameters should be given, default:3 
        patch_size: int[], patch size, default: [7, 3, 3]
        patch_stride: int[], patch_stride ,default: [4, 2, 2]
        patch_padding: int[],patch padding,default: [2, 1, 1]
        embed_dim: int[], mbedding dimension (patch embed out dim), default: [64, 192, 384]
        depth: int[], number ot transformer blocks, default: [1, 2, 10]
        num_heads: int[], number of attention heads, default:[1, 3, 6]
        drop_rate: float[], Mlp layer's droppath rate for droppath layers, default: [0.0, 0.0, 0.0]
        attn_drop_rate: float[], attention layer's droppath rate for droppath layers, default: [0.0, 0.0, 0.0]
        drop_path_rate: float[],each block's droppath rate for droppath layers, default: [0.0, 0.0, 0.1]
        with_cls_token: bool[], if image have cls_token, default: [False, False, True]
    '''

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 num_stage=3,
                 patch_size=[7, 3, 3],
                 patch_stride=[4, 2, 2],
                 patch_padding=[2, 1, 1],
                 embed_dim=[64, 192, 384],
                 depth=[1, 2, 10],
                 num_heads=[1, 3, 6],
                 drop_rate=[0.0, 0.0, 0.0],
                 attn_drop_rate=[0.0, 0.0, 0.0],
                 drop_path_rate=[0.0, 0.0, 0.1],
                 with_cls_token=[False, False, True],
                 ):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = num_stage
        for i in range(self.num_stages):

            stage = VisionTransformer(
                in_chans=in_chans,
                patch_size= patch_size[i],
                patch_stride= patch_stride[i],
                patch_padding= patch_padding[i],
                embed_dim= embed_dim[i],
                depth= depth[i],
                num_heads= num_heads[i],
                mlp_ratio= 4.0,
                qkv_bias= True,
                drop_rate= drop_rate[i],
                attn_drop_rate= attn_drop_rate[i],
                drop_path_rate= drop_path_rate[i],
                with_cls_token= with_cls_token[i],
            )
            setattr(self, f'stage{i}', stage)

            in_chans = embed_dim[i]

        dim_embed = embed_dim[-1]
        self.norm = nn.LayerNorm(dim_embed)
        self.cls_token = with_cls_token[-1]

        # Classifier head
        self.head = nn.Linear(
            dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_init = nn.initializer.TruncatedNormal(std=0.02)
        trunc_init(self.head.weight)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = paddle.load(pretrained, map_location='cpu')
            #logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] is '*'
                )
                if need_init:
                    #if verbose:
                        #logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        #logging.info(
                        #    '=> load_pretrained: resized variant: {} to {}'
                        #    .format(size_pretrained, size_new)
                        #)

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(paddle.sqrt(len(posemb_grid)))
                        gs_new = int(paddle.sqrt(ntok_new))

                        #logging.info(
                        #    '=> load_pretrained: grid-size from {} to {}'
                        #    .format(gs_old, gs_new)
                        #)

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = paddle.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = paddle.to_tensor(
                            paddle.concat([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)
        if self.cls_token:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x)
        else:
            x = graph2vector(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def build_cvt(config):
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        num_stage=config.MODEL.NUM_STAGES,
        patch_size=config.MODEL.PATCH_SIZE,
        patch_stride=config.MODEL.PATCH_STRIDE,
        patch_padding=config.MODEL.PATCH_PADDING,
        embed_dim=config.MODEL.DIM_EMBED,
        depth=config.MODEL.DEPTH,
        num_heads=config.MODEL.NUM_HEADS,
        drop_rate=config.MODEL.DROP_RATE,
        attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        with_cls_token=config.MODEL.CLS_TOKEN
    )
    return model