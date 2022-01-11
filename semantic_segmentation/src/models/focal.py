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
from paddle import nn
from paddle.nn import functional as F

class DropPath(nn.Layer):
    r"""DropPath class"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, set if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        # divide is to keep same output expectation
        output = inputs.divide(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identity(nn.Layer):
    r""" Identity layer
        The output of this layer is the input without any change.
        Use this layer to avoid using 'if' condition in forward methods
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    r""" MLP module
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        weight_attr, bias_attr = self._init_weights()

        self.fc1 = nn.Linear(in_features, hidden_features,
                             weight_attr=weight_attr, bias_attr=bias_attr)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,
                             weight_attr=weight_attr, bias_attr=bias_attr)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr


def window_partition(x, window_size):
    r"""window_partition
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose((0, 1, 3, 2, 4, 5)).reshape((-1, window_size, window_size, C))
    return windows


def window_partition_noreshape(x, window_size):
    r"""window_partition_noreshape
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose((0, 1, 3, 2, 4, 5))
    return windows


def window_reverse(windows, window_size, H, W):
    r"""window_reverse
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape((B, H // window_size, W // window_size, window_size, window_size, -1))
    x = x.transpose((0, 1, 3, 2, 4, 5)).reshape((B, H, W, -1))
    return x


def get_relative_position_index(q_windows, k_windows):
    r"""
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)
    Returns:
        relative_position_index:
            query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = paddle.arange(q_windows[0])
    coords_w_q = paddle.arange(q_windows[1])
    coords_q = paddle.stack(paddle.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = paddle.arange(k_windows[0])
    coords_w_k = paddle.arange(k_windows[1])
    coords_k = paddle.stack(paddle.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = paddle.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = paddle.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    coords_flatten_q = paddle.unsqueeze(coords_flatten_q, axis=-1) # 2, Wh_q*Ww_q, 1
    coords_flatten_k = paddle.unsqueeze(coords_flatten_k, axis=-2) # 2, 1, Ww_k*Ww_k

    relative_coords = coords_flatten_q - coords_flatten_k  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.transpose((1, 2, 0))  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        expand_size (int): The expand size at focal level 1.
        window_size (tuple[int]): The height and width of the window.
        focal_window (int): Focal region size.
        focal_level (int): Focal attention level.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value.
                                    Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pool_method (str): window pooling method. Default: none
    """
    def __init__(self, dim, expand_size, window_size, focal_window,
                    focal_level, num_heads, qkv_bias=True, qk_scale=None,
                    attn_drop=0., proj_drop=0., pool_method="none"):
        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        weight_attr, bias_attr = self._init_weights()

        # define a parameter table of relative position bias for each window
        self.relative_position_bias_table = paddle.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
            dtype=np.float32, is_bias=True)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_l = paddle.unsqueeze(coords_flatten, axis=-1) # 2, Wh*Ww, 1
        coords_flatten_r = paddle.unsqueeze(coords_flatten, axis=-2) # 2, 1, Wh*Ww
        relative_coords = coords_flatten_l - coords_flatten_r  # 2, Wh*Ww, Wh*Ww

        relative_coords = relative_coords.transpose((1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if self.expand_size > 0 and focal_level > 0:
            # define a parameter table of position bias between window
            # and its fine-grained surroundings
            self.window_size_of_key = self.window_size[0] * \
                self.window_size[1] if self.expand_size == 0 else \
                (4 * self.window_size[0] * self.window_size[1] - 4 * \
                (self.window_size[0] -  self.expand_size) * \
                (self.window_size[0] -  self.expand_size))

            self.relative_position_bias_table_to_neighbors = paddle.create_parameter(
                        shape=(1, num_heads,
                        self.window_size[0] * self.window_size[1], self.window_size_of_key),
                        dtype=np.float32, is_bias=True,
                        attr=nn.initializer.TruncatedNormal(std=.02))  # Wh*Ww, nH, nSurrounding

            # get mask for rolled k and rolled v
            mask_tl = paddle.ones((self.window_size[0], self.window_size[1]))
            mask_tl[:-self.expand_size, :-self.expand_size] = 0
            mask_tr = paddle.ones((self.window_size[0], self.window_size[1]))
            mask_tr[:-self.expand_size, self.expand_size:] = 0
            mask_bl = paddle.ones((self.window_size[0], self.window_size[1]))
            mask_bl[self.expand_size:, :-self.expand_size] = 0
            mask_br = paddle.ones((self.window_size[0], self.window_size[1]))
            mask_br[self.expand_size:, self.expand_size:] = 0
            mask_rolled = paddle.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", paddle.flatten(mask_rolled.nonzero()))

        if pool_method != "none" and focal_level > 1:
            self.relative_position_bias_table_to_windows = nn.ParameterList()
            self.unfolds = nn.LayerList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level-1):
                stride = 2**k
                kernel_size = 2*(self.focal_window // 2) + 2**k + (2**k-1)
                # define unfolding operations
                self.unfolds.append(
                    nn.Unfold(
                    kernel_sizes=[kernel_size, kernel_size],
                    strides=stride, paddings=kernel_size // 2)
                )

                # define relative position bias table
                relative_position_bias_table_to_windows = paddle.create_parameter(
                        shape=(self.num_heads,
                        (self.window_size[0] + self.focal_window + 2**k - 2) * \
                        (self.window_size[1] + self.focal_window + 2**k - 2), ),
                        dtype=np.float32, is_bias=True,
                        attr=nn.initializer.TruncatedNormal(std=.02))  # Wh*Ww, nH, nSurrounding
                self.relative_position_bias_table_to_windows.append(
                            relative_position_bias_table_to_windows)

                # define relative position bias index
                relative_position_index_k = get_relative_position_index(self.window_size,
                                            (self.focal_window + 2**k - 1,
                                            self.focal_window + 2**k - 1))
                self.register_buffer("relative_position_index_{}".format(k),
                                                    relative_position_index_k)

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = paddle.zeros(kernel_size, kernel_size)
                    mask[(2**k)-1:, (2**k)-1:] = 1
                    self.register_buffer("valid_ind_unfold_{}".format(k),
                                paddle.flatten(mask.flatten(0).nonzero()))

        self.qkv = nn.Linear(dim, dim * 3, weight_attr=weight_attr,
                             bias_attr=bias_attr if qkv_bias else False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, weight_attr=weight_attr, bias_attr=bias_attr)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x_all (list[Tensors]): input features at different granularity
            mask_all (list[Tensors/None]): masks for input features at different granularity
        """
        x = x_all[0]

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape((B, nH, nW, 3, C)).transpose((3, 0, 1, 2, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C


        # partition q map
        q_windows = window_partition(q, self.window_size[0]).reshape(
                    (-1, self.window_size[0] * self.window_size[0],
                    self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
        k_windows = window_partition(k, self.window_size[0]).reshape(
                    (-1, self.window_size[0] * self.window_size[0],
                    self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
        v_windows = window_partition(v, self.window_size[0]).reshape(
                    (-1, self.window_size[0] * self.window_size[0],
                    self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))

        if self.expand_size > 0 and self.focal_level > 0:
            k_tl = paddle.roll(k, shifts=(-self.expand_size, -self.expand_size), axis=(1, 2))
            v_tl = paddle.roll(v, shifts=(-self.expand_size, -self.expand_size), axis=(1, 2))

            k_tr = paddle.roll(k, shifts=(-self.expand_size, self.expand_size), axis=(1, 2))
            v_tr = paddle.roll(v, shifts=(-self.expand_size, self.expand_size), axis=(1, 2))

            k_bl = paddle.roll(k, shifts=(self.expand_size, -self.expand_size), axis=(1, 2))
            v_bl = paddle.roll(v, shifts=(self.expand_size, -self.expand_size), axis=(1, 2))

            k_br = paddle.roll(k, shifts=(self.expand_size, self.expand_size), axis=(1, 2))
            v_br = paddle.roll(v, shifts=(self.expand_size, self.expand_size), axis=(1, 2))


            k_tl_windows = window_partition(k_tl, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            k_tr_windows = window_partition(k_tr, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            k_bl_windows = window_partition(k_bl, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            k_br_windows = window_partition(k_br, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))

            v_tl_windows = window_partition(v_tl, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            v_tr_windows = window_partition(v_tr, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            v_bl_windows = window_partition(v_bl, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
            v_br_windows = window_partition(v_br, self.window_size[0]).reshape(
            (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))

            k_rolled = paddle.concat((k_tl_windows, k_tr_windows,
                       k_bl_windows, k_br_windows), 1).transpose((0, 2, 1, 3))
            v_rolled = paddle.concat((v_tl_windows, v_tr_windows,
                       v_bl_windows, v_br_windows), 1).transpose((0, 2, 1, 3))

            # mask out tokens in current window
            k_rolled = paddle.gather(k_rolled, self.valid_ind_rolled.flatten(), axis=2)
            v_rolled = paddle.gather(v_rolled, self.valid_ind_rolled.flatten(), axis=2)
            k_rolled = paddle.concat((k_windows, k_rolled), 2)
            v_rolled = paddle.concat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level-1):
                stride = 2**k
                x_window_pooled = x_all[k+1]  # B, nWh, nWw, C
                nWh, nWw = x_window_pooled.shape[1:3] 

                # generate mask for pooled windows
                mask = paddle.ones(shape=(nWh, nWw)).astype(x_window_pooled.dtype)
                unfolded_mask = self.unfolds[k](mask.unsqueeze(0).unsqueeze(1)).reshape((
                    1, 1, self.unfolds[k].kernel_sizes[0],
                    self.unfolds[k].kernel_sizes[1], -1)).transpose((0, 4, 2, 3, 1)).\
                    reshape((nWh*nWw // stride // stride, -1, 1))

                if k > 0:
                    valid_ind_unfold_k = getattr(self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                # from numpy to paddle
                x_window_masks = x_window_masks.numpy()
                x_window_masks[x_window_masks==0] = -100.0
                x_window_masks[x_window_masks>0] = 0.0
                x_window_masks = paddle.to_tensor(x_window_masks.astype(np.float32))         
                mask_all[k+1] = x_window_masks

                # generate k and v for pooled windows                
                qkv_pooled = self.qkv(x_window_pooled).reshape((B, nWh, nWw, 3, C)).transpose(
                                                                              (3, 0, 4, 1, 2))
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]  # B, C, nWh, nWw

                # (B x (nH*nW)) x nHeads x (unfold_wsize x unfold_wsize) x head_dim
                k_pooled_k = self.unfolds[k](k_pooled_k).reshape((
                            B, C, self.unfolds[k].kernel_sizes[0],
                            self.unfolds[k].kernel_sizes[1], -1)).transpose(
                            (0, 4, 2, 3, 1)).reshape((-1,
                            self.unfolds[k].kernel_sizes[0]*self.unfolds[k].kernel_sizes[1],
                            self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
                v_pooled_k = self.unfolds[k](v_pooled_k).reshape((
                            B, C, self.unfolds[k].kernel_sizes[0],
                            self.unfolds[k].kernel_sizes[1], -1)).transpose(
                            (0, 4, 2, 3, 1)).reshape((-1,
                            self.unfolds[k].kernel_sizes[0]*self.unfolds[k].kernel_sizes[1],
                            self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))

                if k > 0:
                    k_pooled_k = k_pooled_k[:, :, valid_ind_unfold_k]
                    v_pooled_k = v_pooled_k[:, :, valid_ind_unfold_k]

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]
            k_all = paddle.concat([k_rolled] + k_pooled, 2)
            v_all = paddle.concat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size
        attn = (paddle.mm(q_windows, k_all.transpose((0, 1, 3, 2))))

        window_area = self.window_size[0] * self.window_size[1]        
        window_area_rolled = k_rolled.shape[2]

        # add relative position bias for tokens inside window
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.flatten()].reshape((
            self.window_size[0] * self.window_size[1], 
            self.window_size[0] * self.window_size[1], -1))
        # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.transpose((2, 0, 1))
        attn[:, :, :window_area, :window_area] = attn[:, :, :window_area, :window_area] + \
                                                 relative_position_bias.unsqueeze(0)

        # add relative position bias for patches inside a window
        if self.expand_size > 0 and self.focal_level > 0:
            attn[:, :, :window_area, window_area:window_area_rolled] = attn[:, :, :window_area,
                window_area:window_area_rolled] + self.relative_position_bias_table_to_neighbors

        if self.pool_method != "none" and self.focal_level > 1:
            # add relative position bias for different windows in an image        
            offset = window_area_rolled
            for k in range(self.focal_level-1):
                # add relative position bias
                relative_position_index_k = getattr(self, 'relative_position_index_{}'.format(k))
                relative_position_bias_to_windows = self.relative_position_bias_table_to_windows[k]
                relative_position_bias_to_windows = paddle.gather(
                    relative_position_bias_to_windows, relative_position_index_k.flatten(),
                    axis=1).reshape((-1, self.window_size[0] * self.window_size[1],
                    (self.focal_window+2**k-1)**2,
                )) # nH, NWh*NWw,focal_region*focal_region
                attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] = \
                    attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] + \
                    relative_position_bias_to_windows.unsqueeze(0)
                # add attentional mask
                if mask_all[k+1] is not None:
                    attn[:, :, :window_area, offset:(offset + (self.focal_window+2**k-1)**2)] = \
                                    attn[:, :, :window_area, offset:(offset + \
                                    (self.focal_window+2**k-1)**2)] + \
                                    paddle.stack([mask_all[k+1].unsqueeze(-2).unsqueeze(-2)] * \
                                    (attn.shape[0] // mask_all[k+1].shape[1]), axis=0).\
                                    reshape((-1, 1, 1, mask_all[k+1].shape[-1]))
                offset += (self.focal_window+2**k-1)**2

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.reshape((attn.shape[0] // nW, nW, self.num_heads, window_area, N))
            attn[:, :, :, :, :window_area] = attn[:, :, :, :, :window_area] + \
                                             mask_all[0].unsqueeze(0).unsqueeze(2)
            attn = attn.reshape((-1, self.num_heads, window_area, N))
            attn = self.softmax(attn)
        else:          
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = paddle.mm(attn, v_all).transpose((0, 2, 1, 3)).reshape(
                                   (attn.shape[0], window_area, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr


class FocalTransformerBlock(nn.Layer):
    r""" Focal Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        expand_size (int): expand size at first focal level (finest level).
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
                                   Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pool_method (str): window pooling method. Default: none, options: [none|fc|conv]
        focal_level (int): number of focal levels. Default: 1.
        focal_window (int): region size of focal attention. Default: 1
        use_layerscale (bool): whether use layer scale for training stability. Default: False
        layerscale_value (float): scaling value for layer scale. Default: 1e-4
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pool_method="none", focal_level=1, focal_window=1, use_layerscale=False,
                 layerscale_value=1e-4):
        super(FocalTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        weight_attr, bias_attr = self._init_weights()

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.LayerList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1,
                                            weight_attr=weight_attr, bias_attr=bias_attr))
                    self.pool_layers[len(self.pool_layers)-1].weight.set_value(
                        paddle.full_like(self.pool_layers[len(self.pool_layers)-1].weight,
                        1./(window_size_glo * window_size_glo))
                    )
                    self.pool_layers[len(self.pool_layers)-1].bias.set_value(
                        paddle.full_like(self.pool_layers[len(self.pool_layers)-1].bias, 0)
                    )
                    
                elif self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2D(dim, dim,
                                            kernel_size=window_size_glo,
                                            stride=window_size_glo, groups=dim))

        self.norm1 = norm_layer(dim,
                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                     bias_attr=bias_attr)

        self.attn = WindowAttention(
            dim, expand_size=self.expand_size,
            window_size=(self.window_size, self.window_size),
            focal_window=focal_window, focal_level=focal_level,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop,proj_drop=drop, pool_method=pool_method)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim,
                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                     bias_attr=bias_attr)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                   act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # from numpy to paddle
            attn_mask = attn_mask.numpy()
            attn_mask[attn_mask!=0] = -100.0
            attn_mask[attn_mask==0] = 0.0
            attn_mask = paddle.to_tensor(attn_mask.astype(np.float32))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        if self.use_layerscale:
            self.gamma_1 = paddle.create_parameter(layerscale_value * paddle.ones((dim)))
            self.gamma_2 = paddle.create_parameter(layerscale_value * paddle.ones((dim)))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape((B, H, W, C))

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, [0, 0, pad_l, pad_r, pad_t, pad_b, 0, 0])

        B, H, W, C = x.shape 

        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                pooled_h = math.ceil(H / self.window_size) * (2 ** k)
                pooled_w = math.ceil(W / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, [0, 0, 0, 0, pad_t, pad_b, 0, 0])

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, [0, 0, pad_l, pad_r, 0, 0, 0, 0])

                # B, nw, nw, window_size, window_size, C
                x_windows_noreshape = window_partition_noreshape(x_level_k, window_size_glo)
                nWh, nWw = x_windows_noreshape.shape[1:3]

                if self.pool_method == "mean":
                    # B, nWh, nWw, C
                    x_windows_pooled = x_windows_noreshape.mean([3, 4])
                elif self.pool_method == "max":
                    # B, nWh, nWw, C
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].reshape(
                                       (B, nWh, nWw, C))
                elif self.pool_method == "fc":
                    # B, nWh, nWw, C, wsize**2
                    x_windows_noreshape = x_windows_noreshape.reshape((B, nWh, nWw,
                                          window_size_glo*window_size_glo, C)).transpose(
                                          (0, 1, 2, 4, 3))
                    # B, nWh, nWw, C
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(-2)
                elif self.pool_method == "conv":
                    # B * nw * nw, C, wsize, wsize
                    x_windows_noreshape = x_windows_noreshape.reshape((-1,
                                          window_size_glo, window_size_glo, C)).transpose(
                                          (0, 3, 1, 2))
                    # B, nWh, nWw, C
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).reshape(
                                       (B, nWh, nWw, C))

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]
        
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)
        attn_windows = attn_windows[:, :self.window_size ** 2]
        
        x = self.merge_windows_and_ffn(attn_windows, shortcut, B, C, H, W)

        return x


    def merge_windows_and_ffn(self, attn_windows, shortcut, B, C, H, W):
        attn_windows = attn_windows.reshape((-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        x = self.reverse_cyclic_shift(shifted_x)
        x = x[:, :self.input_resolution[0], :self.input_resolution[1]].reshape((B, -1, C))

        # FFN
        x = self.ffn(x, shortcut)

        return x


    def reverse_cyclic_shift(self, shifted_x):
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        return x


    def ffn(self, x, shortcut):
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (
                                                  self.gamma_2 * self.mlp(self.norm2(x))))
        return x


    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.
    Args:
        img_size (tuple[int]): Resolution of input feature.
        in_chans (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, img_size, in_chans=3, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.input_resolution = img_size
        self.dim = in_chans
        weight_attr, bias_attr = self._init_weights()
        self.reduction = nn.Linear(4 * in_chans, 2 * in_chans, bias_attr=False)
        self.norm = norm_layer(4 * in_chans,
                    weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                    bias_attr=bias_attr)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape 

        x = x.transpose((0, 2, 3, 1))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape((B, -1, 4 * C))  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr


class BasicLayer(nn.Layer):
    """ A basic Focal Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        expand_size (int): expand size for focal level 1.
        expand_layer (str): expand layer. Default: all
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
                                   Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pool_method (str): Window pooling method. Default: none.
        focal_level (int): Number of focal levels. Default: 1.
        focal_window (int): region size at each focal level. Default: 1.
        use_conv_embed (bool): whether use overlapped convolutional patch embedding layer.
                               Default: False
        use_shift (bool): Whether use window shift as in Swin Transformer. Default: False
        use_pre_norm (bool): Whether use pre-norm before patch embedding projection for stability.
                             Default: False
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
                             Default: None
        use_layerscale (bool): Whether use layer scale for stability. Default: False.
        layerscale_value (float): Layerscale value. Default: 1e-4.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 expand_size, expand_layer="all", mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 pool_method="none", focal_level=1, focal_window=1, use_conv_embed=False,
                 use_shift=False, use_pre_norm=False,downsample=None, use_layerscale=False,
                 layerscale_value=1e-4):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1

        # build blocks
        self.blocks = nn.LayerList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                expand_size=0 if (i % 2 == expand_factor) else expand_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pool_method=pool_method,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim,
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm,
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = x.reshape((x.shape[0], self.input_resolution[0],
                           self.input_resolution[1], -1)).transpose((0, 3, 1, 2))
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Layer):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        use_conv_embed (bool): Wherther use overlapped convolutional embedding layer.
                               Default: False.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_pre_norm (bool): Whether use pre-normalization before projection. Default: False
        is_stem (bool): Whether current patch embedding is stem. Default: False
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96,
                    use_conv_embed=False, norm_layer=None, use_pre_norm=False, is_stem=False):
        super().__init__()
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm

        weight_attr, bias_attr = self._init_weights()

        if use_conv_embed:
            # if we choose to use conv embedding,
            # then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2D(in_chans, embed_dim,
                                 kernel_size=patch_size, stride=patch_size)


        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                bias_attr=bias_attr)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x).flatten(2).transpose((0, 2, 1))  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr


class FocalTransformer(nn.Layer):
    r"""Focal Transformer:Focal Self-attention for Local-Global Interactions in Vision Transformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to
                    the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_shift (bool): Whether to use window shift proposed by Swin Transformer.
                          We observe that using shift or not does not make difference to
                          our Focal Transformer.Default: False
        focal_stages (list): Which stages to perform focal attention.
                             Default: [0, 1, 2, 3], means all stages
        focal_levels (list): How many focal levels at all stages.
                             Note that this excludes the finest-grain level. Default: [1, 1, 1, 1]
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1]
        expand_stages (list): Which stages to expand the finest grain window.
                              Default: [0, 1, 2, 3], means all stages
        expand_sizes (list): The expand size for the finest grain level. Default: [3, 3, 3, 3]
        expand_layer (str): Which layers we want to expand the window for the finest grain leve.
                            This can save computational and memory cost
                            without the loss of performance. Default: "all"
        use_conv_embed (bool): Whether use convolutional embedding.
                               We noted that using convolutional embedding
                               usually improve the performance,
                               but we do not use it by default. Default: False
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False
        layerscale_value (float): Value for layer scale. Default: 1e-4
        use_pre_norm (bool): Whether use pre-norm in patch merging/embedding layer to
                             control the feature magtigute. Default: False
    """
    def __init__(self,
                img_size=224,
                patch_size=4,
                in_chans=3,
                num_classes=1000,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm,
                ape=False,
                patch_norm=True,
                use_shift=False,
                focal_stages=[0, 1, 2, 3],
                focal_levels=[1, 1, 1, 1],
                focal_windows=[7, 5, 3, 1],
                focal_pool="fc",
                expand_stages=[0, 1, 2, 3],
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all",
                use_conv_embed=False,
                use_layerscale=False,
                layerscale_value=1e-4,
                use_pre_norm=False,
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        weight_attr, bias_attr = self._init_weights()

        # split image into patches using either non-overlapped embedding
        # or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_conv_embed=use_conv_embed, is_stem=True,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = paddle.create_parameter(shape=(1, num_patches, embed_dim),
                                      dtype=np.float32, is_bias=True,
                                      attr=nn.initializer.TruncatedNormal(std=.02))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # stochastic depth decay rule
        dpr = [x.numpy().item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                        patches_resolution[1] // (2 ** i_layer)),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    pool_method=focal_pool if i_layer in focal_stages else "none",
                    downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                    focal_level=focal_levels[i_layer],
                    focal_window=focal_windows[i_layer],
                    expand_size=expand_sizes[i_layer],
                    expand_layer=expand_layer,
                    use_conv_embed=use_conv_embed,
                    use_shift=use_shift,
                    use_pre_norm=use_pre_norm,
                    use_layerscale=use_layerscale,
                    layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
            bias_attr=bias_attr)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.head = nn.Linear(self.num_features, num_classes,
            weight_attr=weight_attr, bias_attr=bias_attr) if num_classes > 0 else Identity()


    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        return weight_attr, bias_attr

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table',
                'relative_position_bias_table_to_neighbors',
                'relative_position_bias_table_to_windows'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x
