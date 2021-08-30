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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.upfirdn2d import setup_filter, Upfirdn2dUpsample
from utils.fused_act import fused_leaky_relu


def bias_act(x, b=None, dim=1, gain=None, clamp=None):
    """Slow reference implementation of `bias_act()`
    """
    # spec = activation_funcs[act]
    # alpha = float(alpha if alpha is not None else 0)
    gain = float(gain if gain is not None else 1)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    # alpha = float(alpha)
    # x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clip(-clamp, clamp) # pylint: disable=invalid-unary-operand-type
    return x


def normalize_2nd_moment(x, dim=-1, eps=1e-8):
    return x * (x.square().mean(axis=dim, keepdim=True) + eps).rsqrt()


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


def modulated_style_mlp(x, weight, styles):
    batch_size = x.shape[0]
    channel = x.shape[1]
    width = x.shape[2]
    height = x.shape[3]
    w = None
    dcoefs = None

    w = weight.unsqueeze(0)
    w = w * styles.reshape([batch_size, 1, -1])
    dcoefs = (w.square().sum(axis=[2]) + 1e-8).rsqrt()

    x = x.reshape([batch_size, channel, width * height]).transpose([0, 2, 1])
    x = x * paddle.to_tensor(styles, dtype='float32').reshape([batch_size, 1, -1])
    x = paddle.matmul(x, weight.t())
    x = x * paddle.to_tensor(dcoefs, dtype='float32').reshape([batch_size, 1, -1])
    x = x.transpose([0, 2, 1]).reshape([batch_size, -1, width, height])

    return x


def modulated_channel_attention(x, q_weight, k_weight, v_weight, w_weight,
    u_weight, proj_weight, styles, num_heads):
    """Style modulation effect to the input.
       input feature map is scaled through a style vector,
       which is equivalent to scaling the linear weight.
    """
    batch_size = x.shape[0]
    seq_length = x.shape[1]
    hidden_dimension = x.shape[2]
    depth = hidden_dimension // num_heads
    attention_scale = paddle.to_tensor(depth ** -0.5, dtype='float32')

    layernorm = nn.InstanceNorm1D(seq_length)

    styles1 = styles[:, :hidden_dimension]
    styles2 = styles[:, hidden_dimension:]

    x = x * (styles1.reshape([batch_size, 1, -1]))
    x = layernorm(x)

    q = q_weight.unsqueeze(0)
    q = q * styles1.reshape([batch_size, 1, -1])
    q_dcoefs = (q.square().sum(axis=[2]) + 1e-8).rsqrt()

    k = k_weight.unsqueeze(0)
    k = k * styles1.reshape([batch_size, 1, -1])
    k_dcoefs = (k.square().sum(axis=[2]) + 1e-8).rsqrt()

    v = v_weight.unsqueeze(0)
    v = v * styles1.reshape([batch_size, 1, -1])
    v_dcoefs = (v.square().sum(axis=[2]) + 1e-8).rsqrt()

    w = w_weight.unsqueeze(0)
    w = w * styles2.reshape([batch_size, 1, -1])
    w_dcoefs = (w.square().sum(axis=[2]) + 1e-8).rsqrt()

    q_value = paddle.matmul(x, q_weight.t()) * q_dcoefs.reshape([batch_size, 1, -1])
    q_value = q_value.reshape([batch_size, seq_length, num_heads, depth]).transpose([0, 2, 1, 3])
    k_value = paddle.matmul(x, k_weight.t()) * k_dcoefs.reshape([batch_size, 1, -1])
    k_value = k_value.reshape([batch_size, seq_length, num_heads, depth]).transpose([0, 2, 1, 3])
    if proj_weight is not None:
        k_value = paddle.matmul(k_value.transpose([0, 1, 3, 2]), 
                                proj_weight.t()).transpose([0, 1, 3, 2])
    v_value = paddle.matmul(x, v_weight.t())
    v_value = v_value * v_dcoefs.reshape([batch_size, 1, -1])

    v_value = v_value * styles2.reshape([batch_size, 1, -1])
    skip = v_value
    if proj_weight is not None:
        v_value = paddle.matmul(v_value.transpose([0, 2, 1]), proj_weight.t())
        v_value = v_value.transpose([0, 2, 1])
        v_value = v_value.reshape([batch_size, 256, num_heads, depth]).transpose([0, 2, 1, 3])
    else:
        v_value = v_value.reshape([batch_size, seq_length, num_heads, depth])
        v_value = v_value.transpose([0, 2, 1, 3])

    attn = paddle.matmul(q_value, k_value.transpose([0, 1, 3, 2])) * attention_scale
    revised_attn = attn
    attn_score = F.softmax(revised_attn, axis=-1)

    x = paddle.matmul(attn_score , v_value).transpose([0, 2, 1, 3])
    x = x.reshape([batch_size, seq_length, hidden_dimension])
    x = paddle.matmul(x, paddle.to_tensor(w_weight.t(), dtype='float32'))
    x = x * paddle.to_tensor(w_dcoefs, dtype='float32').reshape([batch_size, 1, -1])

    u = u_weight.unsqueeze(0)
    u = u * styles2.reshape([batch_size, 1, -1])
    u_dcoefs = (u.square().sum(axis=[2]) + 1e-8).rsqrt()

    skip = paddle.matmul(skip, paddle.to_tensor(u_weight.t(), dtype='float32'))
    skip = skip * paddle.to_tensor(u_dcoefs, dtype='float32').reshape([batch_size, 1, -1])

    x = x + skip

    return x


class FullyConnectedLayer(nn.Layer):
    """ FullyConnectedLayer

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Apply additive bias before the activation function
        activation: Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: Learning rate multiplier.
        bias_init: Initial value for the additive bias.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation='linear',
                 lr_multiplier=1,
                 bias_init=0):
        super().__init__()
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.create_parameter(
            shape=[out_features, in_features],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal(std=1e-6))
        self.bias = self.create_parameter(
            shape=[out_features],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = paddle.to_tensor(self.weight, dtype='float32') * self.weight_gain
        b = self.bias
        if b is not None:
            b = paddle.to_tensor(b, dtype='float32')
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = paddle.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = fused_leaky_relu(x, b)
        return x


class MappingNetwork(nn.Layer):
    """ MappingNetwork

    Mapping networks learned affine transformations.

    Attributes:
        z_dim: Input latent (Z) dimensionality, 0 = no latent.
        c_dim: Conditioning label (C) dimensionality, 0 = no label.
        w_dim: Intermediate latent (W) dimensionality.
        num_ws: Number of intermediate latents to output, None = do not broadcast.
        num_layers: Number of mapping layers.
        embed_features: Label embedding dimensionality, None = same as w_dim.
        layer_features: Number of intermediate features in the mapping layers, None = same as w_dim.
        activation: Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: Learning rate multiplier for the mapping layers.
        w_avg_beta: Decay for tracking the moving average of W during training, None = do not track.
    """

    def __init__(self,
                 z_dim,
                 c_dim,
                 w_dim,
                 num_ws,
                 num_layers=2,
                 embed_features=None,
                 layer_features=None,
                 activation='lrelu',
                 lr_multiplier=0.01,
                 w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features,
                                        out_features,
                                        activation=activation,
                                        lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', paddle.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None

        if self.z_dim > 0:
            x = normalize_2nd_moment(paddle.to_tensor(z, dtype='float32'))
        if self.c_dim > 0:
            y = normalize_2nd_moment(paddle.to_tensor(self.embed(c), dtype='float32'))
            x = paddle.concat([x, y], axis=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg = (lerp(x.detach().mean(axis=0), self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).tile([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            if self.num_ws is None or truncation_cutoff is None:
                x = lerp(self.w_avg, x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = lerp(self.w_avg, x[:, :truncation_cutoff], truncation_psi)
        return x


class Encoderlayer(nn.Layer):
    """ Encoderlayer"""
    def __init__(self, h_dim, w_dim, out_dim, seq_length, depth, minimum_head, use_noise=True,
        conv_clamp=None, proj_weight=None, channels_last=False):
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = max(minimum_head, h_dim // depth)
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.use_noise = use_noise
        self.conv_clamp = conv_clamp
        self.affine1 = FullyConnectedLayer(w_dim, h_dim * 2, bias_init=1)

        # memory_format = paddle.channels_last if channels_last else paddle.contiguous_format
        weight_min = -1./math.sqrt(h_dim)
        weight_max = 1./math.sqrt(h_dim)
        self.q_weight = self.create_parameter(
            shape=[h_dim, h_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(weight_min, weight_max))
        self.k_weight = self.create_parameter(
            shape=[h_dim, h_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(weight_min, weight_max))
        self.v_weight = self.create_parameter(
            shape=[h_dim, h_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(weight_min, weight_max))
        self.w_weight = self.create_parameter(
            shape=[out_dim, h_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(weight_min, weight_max))

        self.proj_weight = proj_weight
        self.u_weight = self.create_parameter(
            shape=[out_dim, h_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(weight_min, weight_max))
        if use_noise:
            self.register_buffer('noise_const', paddle.randn([self.seq_length, 1]))
            self.noise_strength = self.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(0.0))
        self.bias = self.create_parameter(
            shape=[out_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x, w, noise_mode='random', gain=1):
        styles1 = self.affine1(w)
        noise = None

        if self.use_noise and noise_mode == 'random':
            noise = paddle.randn([x.shape[0], self.seq_length, 1]) * self.noise_strength[0]
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength[0]

        x = modulated_channel_attention(x=x, q_weight=self.q_weight, k_weight=self.k_weight,
            v_weight=self.v_weight, w_weight=self.w_weight, u_weight=self.u_weight,
            proj_weight=self.proj_weight, styles=styles1, num_heads=self.num_heads)

        if noise is not None:
            x = x.add_(noise)

        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None

        x = x + paddle.to_tensor(self.bias, dtype='float32')
        x = F.leaky_relu(x, negative_slope=0.2)
        x = paddle.clip(x, max=act_clamp, min=-act_clamp)

        return x


class ToRGBLayer(nn.Layer):
    """ToRGBLayer

    Convert reshaped output for each resolution into an RGB channel.

    """

    def __init__(self, in_channels, out_channels, w_dim, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = None
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # memory_format = paddle.channels_last if channels_last else paddle.contiguous_format
        self.weight = self.create_parameter(
            shape=[out_channels, in_channels],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Uniform(
                -1./math.sqrt(in_channels), 1./math.sqrt(in_channels)))
        self.bias = self.create_parameter(
            shape=[out_channels],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w)
        x = modulated_style_mlp(x=x, weight=self.weight, styles=styles)

        x = bias_act(x, self.bias, clamp=self.conv_clamp)

        return x


class EncoderBlock(nn.Layer):
    """EncoderBlock

    Attributes:
        w_dim: Intermediate latent (W) dimensionality.
        img_resolution: int, size of image
        img_channels: int, channel of input image
    """

    def __init__(self, h_dim, w_dim, out_dim, depth, minimum_head, img_resolution, resolution,
        img_channels, is_first, is_last, init_resolution, architecture='skip', linformer=False,
        conv_clamp=None, use_fp16=False, fp16_channels_last=False, resample_filter =[1,3,3,1],
        scale_ratio=2):
        super().__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.depth = depth
        self.minimum_head = minimum_head
        self.img_resolution = img_resolution
        self.init_resolution = init_resolution
        self.resolution = resolution
        self.img_channels = img_channels
        self.seq_length = resolution * resolution
        self.is_first = is_first
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.register_buffer('resample_filter', setup_filter(resample_filter))
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.num_attention = 0
        self.num_torgb = 0
        self.scale_ratio = scale_ratio
        self.conv_clamp = conv_clamp
        self.proj_weight = None

        # memory_format = paddle.contiguous_format

        if self.resolution>=32 and linformer:
            self.proj_weight = self.create_parameter(
                shape=[256, self.seq_length],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Uniform(
                    -1./math.sqrt(self.seq_length), 1./math.sqrt(self.seq_length)))

        if self.resolution == self.init_resolution and self.is_first:
            self.const = self.create_parameter(
                shape=[self.seq_length, self.h_dim],
                dtype='float32',
                default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        if self.is_first:
            self.pos_embedding = self.create_parameter(
                shape=[1, self.seq_length, self.h_dim],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(0))

        if not self.is_last or out_dim is None:
            self.out_dim = h_dim

        self.enc = Encoderlayer(h_dim=self.h_dim, w_dim=self.w_dim, out_dim=self.out_dim,
            seq_length=self.seq_length, depth=self.depth, minimum_head=self.minimum_head,
            conv_clamp=self.conv_clamp, proj_weight=self.proj_weight)
        self.num_attention += 1

        if self.is_last and self.architecture == 'skip':
            self.torgb = ToRGBLayer(self.out_dim, self.img_channels, w_dim=w_dim,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

    def forward(self, x, img, ws, force_fp32=True, fused_modconv=None):
        w_iter = iter(ws.unbind(axis=1))
        # memory_format = paddle.channels_last if self.channels_last and not force_fp32 else paddle.contiguous_format
        # if fused_modconv is None:
        #     fused_modconv = (not self.training) and (fused_modconv.dtype == 'float32' or int(x.shape[0]) == 1)

        #Input
        if self.is_first and self.resolution == self.init_resolution:
            x = paddle.to_tensor(self.const, dtype='float32')
            x = x.unsqueeze(0).tile([ws.shape[0], 1, 1])
        else:
            x = paddle.to_tensor(x, dtype='float32')

        #Main layers
        if self.is_first:
            x = x + self.pos_embedding

        if self.architecture == 'resnet':
            y = self.skip(x.transpose([0,2,1]).reshape(
                [ws.shape[0], self.h_dim, self.resolution, self.resolution]))
            x = self.enc(x, next(w_iter))
            y = y.reshape([ws.shape[0], self.h_dim, self.seq_length])
            x = y.add_(x)
        else:
            x = paddle.to_tensor(self.enc(x, next(w_iter)))

        #ToRGB
        if self.is_last:
            if img is not None:
                upsample2d = Upfirdn2dUpsample(self.resample_filter)
                img = upsample2d(img)

            if self.architecture == 'skip':
                y = self.torgb(x.transpose([0,2,1]).reshape(
                    [ws.shape[0], self.out_dim, self.resolution, self.resolution]),
                    next(w_iter),
                    fused_modconv=fused_modconv)
                y = paddle.to_tensor(y, dtype='float32')
                img = img.add_(y) if img is not None else y

            #upsample
            if self.resolution!=self.img_resolution:
                upsample2d = Upfirdn2dUpsample(self.resample_filter)
                x = upsample2d(x.transpose([0,2,1]).reshape([ws.shape[0],
                               self.out_dim, self.resolution, self.resolution]))
                x = x.reshape([ws.shape[0], self.out_dim, self.seq_length * self.scale_ratio **2])
                x = x.transpose([0,2,1])

        return x, img


class SynthesisNetwork(nn.Layer):
    """SynthesisNetwork

    Attributes:
        w_dim: Intermediate latent (W) dimensionality.
        img_resolution: int, size of image
        img_channels: int, channel of input image
        num_block: int, Number of layers
        num_ws: Number of intermediate latents to output, None = do not broadcast.
    """
    def __init__(self, w_dim, img_resolution, img_channels, depth, num_layers, G_dict,
        linformer, init_resolution, minimum_head=1, conv_clamp=256, num_fp16_res=0):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_block = num_layers
        self.linformer = linformer
        if init_resolution==12:
            self.block_resolutions = [3 * 2 ** i for i in range(2, self.img_resolution_log2)]
        else:
            self.block_resolutions = [2 ** i for i in range(3, self.img_resolution_log2 + 1)]

        channels_dict = dict(zip(*[self.block_resolutions, G_dict]))
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for i, res in enumerate(self.block_resolutions):
            h_dim = channels_dict[res]
            out_dim = None
            if res!=self.img_resolution:
                out_dim = channels_dict[res*2]
            use_fp16 = (res >= fp16_resolution)
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                is_first = (j == 0)
                is_last = (j == num_block_res - 1)
                block = EncoderBlock(
                    h_dim=h_dim, w_dim=w_dim, out_dim=out_dim, depth=depth,
                    minimum_head=minimum_head, img_resolution=img_resolution,
                    resolution=res, img_channels=img_channels, is_first=is_first,
                    is_last=is_last, use_fp16=use_fp16, conv_clamp=conv_clamp,
                    linformer=self.linformer, init_resolution=init_resolution)
                self.num_ws += block.num_attention
                if is_last:
                    self.num_ws += block.num_torgb
                setattr(self, f'b{res}_{j}', block)

    def forward(self, ws=None):
        block_ws = []
        ws = paddle.to_tensor(ws, dtype='float32')
        w_idx = 0
        for i, res in enumerate(self.block_resolutions):
            num_block_res = self.num_block[i]
            res_ws = []
            for j in range(num_block_res):
                block = getattr(self, f'b{res}_{j}')
                res_ws.append(ws.slice(axes=[1], starts=[w_idx],
                              ends=[w_idx + block.num_attention + block.num_torgb]))
                w_idx += block.num_attention
            block_ws.append(res_ws)

        x = img = None
        for i, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            num_block_res = self.num_block[i]
            for j in range(num_block_res):
                block = getattr(self, f'b{res}_{j}')
                x, img = block(x, img, cur_ws[j])

        return img


class Generator(nn.Layer):
    """Generator class

    Attributes:
        z_dim: Input latent (Z) dimensionality, 0 = no latent.
        c_dim: Conditioning label (C) dimensionality, 0 = no label.
        w_dim: Intermediate latent (W) dimensionality.
        img_resolution: int, size of image
        img_channels: int, channel of input image
        num_ws: Number of intermediate latents to output, None = do not broadcast.
    """

    def __init__(self, config):
        super().__init__()
        self.img_resolution = config.DATA.IMAGE_SIZE
        self.img_channels = config.DATA.CHANNEL
        self.z_dim = config.MODEL.GEN.Z_DIM
        self.c_dim = config.MODEL.GEN.C_DIM
        self.w_dim = config.MODEL.GEN.W_DIM
        self.depth = config.MODEL.GEN.DEPTH
        self.num_layers = config.MODEL.GEN.NUM_LAYERS
        self.G_dict = config.MODEL.GEN.G_DICT
        self.linformer = config.MODEL.GEN.LINFORMER
        self.init_resolution = config.MODEL.GEN.RESOLUTION
        self.synthesis = SynthesisNetwork(w_dim=self.w_dim, img_resolution=self.img_resolution,
            depth=self.depth, num_layers=self.num_layers, G_dict=self.G_dict,
            img_channels=self.img_channels, linformer=self.linformer,
            init_resolution=self.init_resolution)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim,
                                      w_dim=self.w_dim, num_ws=self.num_ws)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        output = self.synthesis(ws)

        return output
