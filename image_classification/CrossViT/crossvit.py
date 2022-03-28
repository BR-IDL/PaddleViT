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
CrossViT in Paddle
A Paddle Implementation of Cross-Attention Multi-Scale Vision Transformer (CrossViT) as described in:
"CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification"
    - Paper Link: https://arxiv.org/abs/2103.14899
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial
from t2t import T2T, get_sinusoid_encoding
from crossvit_utils import *

class PatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2D(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2D(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2D(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2D(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv2D(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2D(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose((0, 2, 1))

        return x


class CrossAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.wq = nn.Linear(dim, dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.wk = nn.Linear(dim, dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        w_attr_3, b_attr_3 = self._init_weights()
        self.wv = nn.Linear(dim, dim, weight_attr=w_attr_3, bias_attr=b_attr_3)
        self.attn_drop = nn.Dropout(attn_drop)
        w_attr_4, b_attr_4 = self._init_weights()
        self.proj = nn.Linear(dim, dim, weight_attr=w_attr_4, bias_attr=b_attr_4)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, :]).reshape([B, 1, self.num_heads, C // self.num_heads]).transpose(
            (0, 2, 1, 3))  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose(
            (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose(
            (0, 2, 1, 3))  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose((0, 1, 3, 2))) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = (attn @ v).transpose((0, 2, 1, 3)).reshape([B, 1, C])  
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 has_mlp=True):
        super(CrossAttentionBlock, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.norm1 = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)
        self.attn = CrossAttention(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   attn_drop=attn_drop,
                                   proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, dropout=drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        x = x[:, 0:1, :] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Layer):
    def __init__(self,
                 dim,
                 patches,
                 depth,
                 num_heads,
                 mlp_ratio,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=[]):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.LayerList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d],
                          num_heads=num_heads[d],
                          mlp_ratio=mlp_ratio[d],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          dropout=drop,
                          attention_dropout=attn_drop,
                          droppath=drop_path[i]))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.LayerList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [Identity()]
            else:
                w_attr_1, b_attr_1 = self._init_weights_norm()
                w_attr_2, b_attr_2 = self._init_weights_linear()
                tmp = [nn.LayerNorm(dim[d], weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6),
                       nn.GELU(),
                       nn.Linear(dim[d],
                                 dim[(d + 1) % num_branches],
                                 weight_attr=w_attr_2,
                                 bias_attr=b_attr_2)]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.LayerList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(dim=dim[d_],
                                        num_heads=nh,
                                        mlp_ratio=mlp_ratio[d],
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop,
                                        attn_drop=attn_drop,
                                        drop_path=drop_path[-1],
                                        has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_],
                                                   num_heads=nh,
                                                   mlp_ratio=mlp_ratio[d],
                                                   qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop,
                                                   attn_drop=attn_drop,
                                                   drop_path=drop_path[-1],
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.LayerList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [Identity()]
            else:
                w_attr_1, b_attr_1 = self._init_weights_norm()
                w_attr_2, b_attr_2 = self._init_weights_linear()
                tmp = [nn.LayerNorm(dim[(d + 1) % num_branches],
                                    weight_attr=w_attr_1,
                                    bias_attr=w_attr_1),
                       nn.GELU(),
                       nn.Linear(dim[(d + 1) % num_branches],
                                 dim[d],
                                 weight_attr=w_attr_2,
                                 bias_attr=b_attr_2)]
            self.revert_projs.append(nn.Sequential(*tmp))

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = paddle.concat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, :]), axis=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, :])
            tmp = paddle.concat((reverted_proj_cls_token, outs_b[i][:, 1:, :]), axis=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size, patches)]


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=(224, 224),
                 patch_size=(8, 16),
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12),
                 mlp_ratio=(2., 2., 4.),
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 hybrid_backbone=None,
                 multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)
        self.patch_embed = nn.LayerList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList(
                [paddle.create_parameter(
                    shape=[1, 1 + num_patches[i], embed_dim[i]],
                    dtype='float32',
                    default_initializer=nn.initializer.Constant(
                        0.0)) for i in range(self.num_branches)])

            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(
                    PatchEmbed(img_size=im_s,
                               patch_size=p,
                               in_chans=in_chans,
                               embed_dim=d,
                               multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(
                    T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(
                    paddle.to_tensor(data=get_sinusoid_encoding(n_position=1 + num_patches[idx],
                                                                d_hid=embed_dim[idx]),
                                     dtype='flaot32',
                                     stop_gradient=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList(
                [paddle.to_tensor(
                    paddle.zeros(1, 1 + num_patches[i], embed_dim[i]),
                    dtype='float32',
                    stop_gradient=False) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList(
            [paddle.create_parameter(
                shape=[1, 1, embed_dim[i]], dtype='float32') for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=dropout)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in paddle.linspace(0, droppath, total_depth)]
        dpr_ptr = 0
        self.blocks = nn.LayerList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim,
                                  num_patches,
                                  block_cfg,
                                  num_heads=num_heads,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  qk_scale=qk_scale,
                                  drop=dropout,
                                  attn_drop=attention_dropout,
                                  drop_path=dpr_)
            dpr_ptr += curr_depth
            self.blocks.append(blk)
    
        
        w_attr_1, b_attr_1 = self._init_weights_norm()
        w_attr_2, b_attr_2 = self._init_weights_linear()
        self.norm = nn.LayerList([nn.LayerNorm(embed_dim[i],
            weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6) for i in range(self.num_branches)])
        self.head = nn.LayerList(
            [nn.Linear(embed_dim[i],
                       num_classes,
                       weight_attr=w_attr_2,
                       bias_attr=b_attr_2) if num_classes > 0 else Identity() for i in range(self.num_branches)])

    def _init_weights_norm(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def _init_weights_linear(self):
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0))
        return weight_attr, bias_attr

    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = paddle.nn.functional.interpolate(
                x, size=(self.img_size[i],
                         self.img_size[i]),
                         mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand([B, -1, -1])  # stole cls_tokens impl from Phil Wang, thanks
            # print(cls_tokens.shape,tmp.shape)
            tmp = paddle.concat((cls_tokens, tmp), axis=1)
            # print(tmp.shape,self.pos_embed[i].shape)
            tmp = tmp+self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)
            # print(xs.shape)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def forward(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = paddle.mean(paddle.stack(ce_logits, axis=0), axis=0)
        return ce_logits


def build_crossvit(config):
    """build corssvit model using config"""
    model = VisionTransformer(img_size=config.MODEL.IMG_SIZE,
                              num_classes=config.MODEL.NUM_CLASSES,
                              patch_size=config.MODEL.PATCH_SIZE,
                              embed_dim=config.MODEL.EMBED_DIM,
                              depth=config.MODEL.DEPTH,
                              num_heads=config.MODEL.NUM_HEADS,
                              mlp_ratio=config.MODEL.MLP_RATIO,
                              qkv_bias=config.MODEL.QKV_BIAS,
                              multi_conv=config.MODEL.MULTI_CONV,
                              hybrid_backbone=config.MODEL.HYBRID_BACKBONE,
                              dropout=config.MODEL.DROPOUT,
                              attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                              droppath=config.MODEL.DROPPATH)
    return model
