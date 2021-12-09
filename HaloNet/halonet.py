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
Implement Network Class for HaloNet
"""

import paddle
from paddle import nn
paddle.set_device('cpu')


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    """ calculate new vector dim according to input vector dim
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v

def init_weights():
    """ init Linear weight
    """
    weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=.02))
    bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
    return weight_attr, bias_attr


class Identity(nn.Layer):
    """ Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvBnAct(nn.Layer):
    """ Build layer contain: conv - bn - act
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 act=None,
                 bias_attr=False,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(
                                  initializer=nn.initializer.KaimingUniform()),
                              bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_channels)

        self.act = ActLayer(act)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.act(out)
        return out


class BatchNormAct2d(nn.Layer):
    """ Build layer contain: bn-act
    """
    def __init__(self, chs, act=None):
        super().__init__()
        self.bn = nn.BatchNorm2D(chs)
        self.act = ActLayer(act)

    def forward(self, inputs):
        out = self.bn(inputs)
        out = self.act(out)
        return out


class ActLayer(nn.Layer):
    """ Build Activation Layer according to act type
    """
    def __init__(self, act):
        super().__init__()
        if act == 'silu':
            self.act = nn.Silu()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = Identity()

    def forward(self, x):
        out = self.act(x)
        return out


class SelectAdaptivePool2d(nn.Layer):
    """ Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super().__init__()
        # convert other false values to empty string for consistent TS typing
        self.pool_type = pool_type or ''
        self.flatten = nn.Flatten(1) if flatten else Identity()
        if pool_type == '':
            self.pool = Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2D(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x


class Stem(nn.Layer):
    def __init__(self, act):
        super().__init__()

        self.conv1 = ConvBnAct(3, 24, kernel_size=3, stride=2, padding=1, act=act)
        self.conv2 = ConvBnAct(24, 32, kernel_size=3, stride=1, padding=1, act=act)
        self.conv3 = ConvBnAct(32, 64, kernel_size=3, stride=1, padding=1, act=act)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ceil_mode=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x


def rel_logits_1d(q, rel_k, permute_mask):
    """ Compute relative logits along one dimension
    :param q: [batch,H,W,dim]
    :param rel_k: [2*window-1,dim]
    :param permute_mask: permute output axis according to this
    """

    B, H, W, _ = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size+1)//2

    rel_k = paddle.transpose(rel_k, [1, 0])
    x = (q@rel_k)
    x = x.reshape([-1, W, rel_size])

    # pad to shift from relative to absolute indexing
    x_pad = paddle.nn.functional.pad(x, [0, 1],data_format='NCL')
    x_pad = x_pad.flatten(1)
    x_pad = x_pad.unsqueeze(1)
    x_pad = paddle.nn.functional.pad(x_pad, [0, rel_size - W], data_format='NCL')
    x_pad = x_pad.squeeze()

    # reshape adn slice out the padded elements
    x_pad = x_pad.reshape([-1, W+1, rel_size])    #[25088,9,27]
    x = x_pad[:, :W, win_size-1:]     # [25088,8,14]

    # reshape and tile
    x = x.reshape([B, H, 1, W, win_size])
    x = x.expand([-1, -1, win_size, -1, -1])
    x = paddle.transpose(x, permute_mask)

    return x


class RelPosEmb(nn.Layer):
    """ Relative Position Embedding
    """
    def __init__(self,
                 block_size,
                 win_size,
                 dim_head,
                 ):
        """
        :param block_size (int): block size
        :param win_size (int): neighbourhood window size
        :param dim_head (int): attention head dim
        :param scale (float): scale factor (for init)
        """
        super().__init__()

        self.block_size = block_size
        self.dim_head = dim_head

        self.rel_height = paddle.create_parameter(
            shape=[(2 * win_size - 1), dim_head],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        self.rel_width = paddle.create_parameter(
            shape=[(2 * win_size - 1), dim_head],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

    def forward(self, q):
        B,BB,HW,_ = q.shape

        # relative logits in width dimension
        q = q.reshape([-1,self.block_size,self.block_size,self.dim_head])
        rel_logits_w = rel_logits_1d(q,self.rel_width,permute_mask=[0,1,3,2,4])

        # relative logits in height dimension
        q = paddle.transpose(q,[0,2,1,3])
        rel_logits_h = rel_logits_1d(q,self.rel_height,permute_mask=[0,3,1,4,2])

        rel_logits = rel_logits_h+rel_logits_w
        rel_logits = rel_logits.reshape([B,BB,HW,-1])

        return rel_logits


class HaloAttention(nn.Layer):
    """
    The internal dimensions of the attention module are controlled by
    the interaction of several arguments.
    the output dimension : dim_out
    the value(v) dimension :  dim_out//num_heads
    the query(q) and key(k) dimensions are determined by :
         * num_heads*dim_head
         * num_heads*(dim_out*attn_ratio//num_heads)
    the ratio of q and k relative to the output : attn_ratio

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """
    def __init__(self,
                 dim,
                 dim_out=None,
                 feat_size=None,
                 stride=1,
                 num_heads=8,
                 dim_head=None,
                 block_size=8,
                 halo_size=3,
                 qk_ratio=1.0,
                 qkv_bias=False,
                 avg_down=False,
                 scale_pos_embed=False):

        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.stride = stride
        self.num_heads = num_heads
        self.dim_head_qk = make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = stride
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = stride
            self.block_size_ds = self.block_size // self.block_stride
        self.q = nn.Conv2D(dim,
                           self.dim_out_qk,
                           1,
                           stride=self.block_stride,
                           bias_attr=qkv_bias,
                           weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))
        self.kv = nn.Conv2D(dim, self.dim_out_qk + self.dim_out_v, 1, bias_attr=qkv_bias)
        self.pos_embed = RelPosEmb(
            block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk)
        self.pool = nn.AvgPool2D(2, 2) if use_avg_pool else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0 and W % self.block_size == 0, 'fmap dimensions must be divisible by the block size'
        num_h_blocks = H//self.block_size
        num_w_blocks = W//self.block_size
        num_blocks = num_h_blocks * num_w_blocks

        q = self.q(x)
        # unfold
        q = q.reshape([-1,self.dim_head_qk,num_h_blocks,self.block_size_ds,num_w_blocks,self.block_size_ds])
        q = paddle.transpose(q,[0,1,3,5,2,4])
        q = q.reshape([B*self.num_heads,self.dim_head_qk,-1,num_blocks])
        q = paddle.transpose(q,[0,3,2,1])  # B*num_heads,num_blocks,block_size**2, dim_head
        kv = self.kv(x)

        # # generate overlap windows for kv---solution 1 unfold.unfold
        # kv = paddle.nn.functional.pad(kv, [self.halo_size, self.halo_size, self.halo_size, self.halo_size]) # [bs,dim_out,pad_H,pad_W]
        # kv = self.kv_unfold(kv)  # dimension=2

        # # another function to generate overlap windows for kv---solution 2 is xla
        # WW = self.win_size ** 2
        # pw = paddle.eye(WW, dtype=x.dtype).reshape([WW, 1, self.win_size, self.win_size])
        # kv = paddle.nn.functional.conv2d(kv.reshape([-1, 1, H, W]), pw, stride=self.block_size, padding=self.halo_size)

        # the other function to generate overlap between windows for kv --- solution 3
        kv_unfold = nn.Unfold([self.win_size,self.win_size], strides=self.block_size, paddings=self.halo_size)
        kv = kv_unfold(kv)

        kv = kv.reshape([B * self.num_heads, self.dim_head_qk + self.dim_head_v, -1, num_blocks,])
        kv = paddle.transpose(kv,[0, 3, 2, 1])

        k, v = paddle.split(kv, [self.dim_head_qk, self.dim_head_v], axis=-1)
        k = paddle.transpose(k,[0,1,3,2])

        if self.scale_pos_embed:
            attn = (q@k + self.pos_embed(q)) * self.scale
        else:
            pos_embed_q = self.pos_embed(q)
            part_1 = (q @ k) * self.scale
            attn = part_1 + pos_embed_q
        # attn: B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        softmax_fn = nn.layer.Softmax(-1)
        attn = softmax_fn(attn)
        attn = attn @ v

        out = paddle.transpose(attn,[0,3,2,1])  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = out.reshape([-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks])
        out = paddle.transpose(out,[0, 3, 1, 4, 2])
        out = out.reshape(
            [B, self.dim_out_v, H // self.block_stride, W // self.block_stride])
        # B, dim_out, H // block_stride, W // block_stride
        out = self.pool(out)
        return out


class BottleneckBlock(nn.Layer):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride,
                 act,
                 downsample=None,
                 shortcut=None,
                 ):
        super().__init__()

        self.stride = stride
        mid_chs = out_chs//4

        self.conv1_1x1 = ConvBnAct(in_chs,
                                   mid_chs,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   act=act)
        self.conv2_kxk = ConvBnAct(mid_chs,
                                   mid_chs,
                                   kernel_size=3,
                                   stride=self.stride,
                                   padding=1,
                                   act=act)
        self.conv2b_kxk = Identity()
        self.conv3_1x1 = ConvBnAct(mid_chs,
                                   out_chs,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.attn = Identity()
        self.attn_last = Identity()
        self.shortcut = shortcut

        if self.shortcut:
            if downsample:
                self.creat_shortcut = ConvBnAct(in_chs,
                                                out_chs,
                                                kernel_size=1,
                                                stride=self.stride,
                                                padding=0)
            else:
                self.creat_shortcut = ConvBnAct(in_chs,
                                                out_chs,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

        self.Identity = Identity()
        self.act = ActLayer(act)

    def forward(self, x):
        h = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.conv2b_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        out = self.attn_last(x)
        if self.shortcut:
            h = self.creat_shortcut(h)
        else:
            h = self.Identity(h)
        out = out + h
        out = self.act(out)
        return out


class SelfAttnBlock(nn.Layer):
    """ ResNet-like Bottleneck Block - 1x1 -kxk - self attn -1x1
    """
    def __init__(self,
                 chs,
                 num_heads,
                 block_size,
                 halo_size,
                 act,
                 stride=None,
                 shortcut=None,
                 hidden_chs=None,
                 ):
        super().__init__()
        mid_chs = chs//4

        if hidden_chs is None:
            out_chs = chs
        else:
            out_chs = hidden_chs

        if stride is None:
            self.stride = 1
        else:
            self.stride = stride

        self.conv1_1x1 = ConvBnAct(out_chs, mid_chs, kernel_size=1, stride=1, padding=0,act=act)
        self.conv2_kxk = Identity()
        self.conv3_1x1 = ConvBnAct(mid_chs, chs, kernel_size=1, stride=1, padding=0)

        self.self_attn = HaloAttention(mid_chs,
                                       dim_out=mid_chs,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       num_heads=num_heads,
                                       stride=self.stride)
        self.post_attn = BatchNormAct2d(mid_chs,act=act)

        self.shortcut = shortcut
        if self.shortcut:
            self.creat_shortcut = ConvBnAct(out_chs,
                                            chs,
                                            kernel_size=1,
                                            stride=self.stride,
                                            padding=0)
        self.Identity = Identity()
        self.act = ActLayer(act=act)

    def forward(self, x):
        h = x
        out = self.conv1_1x1(x)
        out = self.self_attn(out)
        out = self.post_attn(out)
        out = self.conv3_1x1(out)
        if self.shortcut:
            h = self.creat_shortcut(h)
        else:
            h = self.Identity(h)
        out = out + h
        out = self.act(out)
        return out


class HaloStage(nn.Layer):
    """ Stage layers for HaloNet. Stage layers contains a number of Blocks.
    """
    def __init__(self,
                 block_types,
                 block_size,
                 halo_size,
                 depth,
                 channel,
                 out_channel,
                 stride,
                 num_head,
                 act,
                 hidden_chs=None,
                 downsample=None,
                 ):
        super().__init__()

        self.depth = depth

        blocks = []

        for idx in range(depth):
            if idx == 0:
                shortcut = True
                in_channel = channel
                if downsample is None:
                    self.down = False
                else:
                    self.down = downsample
                block_stride = stride
                self.hidden = hidden_chs
            else:
                stride = 1
                shortcut = False
                in_channel = out_channel
                self.down = False
                block_stride = 1
                self.hidden = None

            block_type = block_types[idx]
            if block_type == 'bottle':
                blocks.append(
                    BottleneckBlock(
                        in_chs=in_channel,
                        out_chs=out_channel,
                        stride=block_stride,
                        shortcut=shortcut,
                        downsample=self.down,
                        act=act,
                    )
                )

            if block_type == 'attn':
                if num_head > 0:
                    blocks.append(
                        SelfAttnBlock(
                            chs=out_channel,
                            stride=stride,
                            num_heads=num_head,
                            block_size=block_size,
                            halo_size=halo_size,
                            hidden_chs=self.hidden,
                            shortcut=shortcut,
                            act=act,
                        )
                )

        self.blocks = nn.LayerList(blocks)

    def forward(self, x):
        for stage in self.blocks:
            x = stage(x)
        return x


class HaloNet(nn.Layer):
    """ Define main structure of HaloNet: stem - blocks - head
    """
    def __init__(self,
                 depth_list,
                 block_size,
                 halo_size,
                 stage1_block,
                 stage2_block,
                 stage3_block,
                 stage4_block,
                 chs_list,
                 num_heads,
                 num_classes,
                 stride_list,
                 hidden_chs,
                 act,
                 ):
        super().__init__()
        self.stem = Stem(act)
        self.stage1 = HaloStage(
                                block_types=stage1_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[0],
                                channel=chs_list[0],
                                out_channel=chs_list[1],
                                stride=stride_list[0],
                                num_head=num_heads[0],
                                hidden_chs=hidden_chs,
                                act=act,
                                )
        self.stage2 = HaloStage(
                                block_types=stage2_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[1],
                                channel=chs_list[1],
                                out_channel=chs_list[2],
                                stride=stride_list[1],
                                num_head=num_heads[1],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)
        self.stage3 = HaloStage(
                                block_types=stage3_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[2],
                                channel=chs_list[2],
                                out_channel=chs_list[3],
                                stride=stride_list[2],
                                num_head=num_heads[2],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)
        self.stage4 = HaloStage(
                                block_types=stage4_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[3],
                                channel=chs_list[3],
                                out_channel=chs_list[4],
                                stride=stride_list[3],
                                num_head=num_heads[3],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)

        w_attr_1, b_attr_1 = init_weights()
        self.classifier = nn.Sequential(
            SelectAdaptivePool2d(flatten=True),
            nn.Linear(chs_list[4], num_classes, weight_attr=w_attr_1, bias_attr=b_attr_1),
            Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        out_stage1 = self.stage1(x)
        out_stage2 = self.stage2(out_stage1)
        out_stage3 = self.stage3(out_stage2)
        out_stage4 = self.stage4(out_stage3)
        out = self.classifier(out_stage4)
        return out


def build_halonet(config):
    """ Build HaloNet by reading options in config object
    :param config: config instance contains setting options
    :return: HaloNet model
    """
    model = HaloNet(
                    depth_list=config.MODEL.DEPTH,
                    stage1_block=config.MODEL.STAGE1_BLOCK,
                    stage2_block=config.MODEL.STAGE2_BLOCK,
                    stage3_block=config.MODEL.STAGE3_BLOCK,
                    stage4_block=config.MODEL.STAGE4_BLOCK,
                    chs_list=config.MODEL.CHANNEL,
                    num_heads=config.MODEL.NUM_HEAD,
                    num_classes=config.MODEL.NUM_CLS,
                    stride_list=config.MODEL.STRIDE,
                    block_size=config.MODEL.BLOCK_SIZE,
                    halo_size=config.MODEL.HALO_SIZE,
                    hidden_chs=config.MODEL.HIDDEN_CHANNEL,
                    act=config.MODEL.ACT,
    )
    return model
