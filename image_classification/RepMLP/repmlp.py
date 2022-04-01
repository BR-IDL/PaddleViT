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
RepMLP in Paddle
A Paddle Implementation of RepMLP as described in:
"RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition"
    - Paper Link: https://arxiv.org/abs/2112.11081
"""

import copy
import paddle
import paddle.nn.functional as F
from paddle import nn



def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1, relu=True):
    ops = [] 
    ops.append(('conv', nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias_attr=False)))
    ops.append(('bn', nn.BatchNorm2D(num_features=out_channels)))
    if relu is True:
        ops.append(('relu', nn.ReLU()))
    return nn.Sequential(*ops)


def fuse_bn(conv_or_fc, bn):
    std = (bn._variance + bn.epsilon).sqrt()
    t = bn.weight / std
    t = t.reshape([-1, 1, 1, 1])

    if len(t) == conv_or_fc.weight.shape[0]:
        return conv_or_fc.weight * t, bn.bias - bn._mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.shape[0] // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn._mean * bn.weight / std).repeat_interleave(
            repeat_times, 0)


class Identity(nn.Layer):
    def forward(self, x):
        return x


class GlobalPerceptron(nn.Layer):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2D(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias_attr=True)
        self.fc2 = nn.Conv2D(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias_attr=True)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_channels = input_channels

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.reshape([-1, self.input_channels, 1, 1])
        return x


class RepMLPBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 h,
                 w,
                 reparam_conv_k=None,
                 globalperceptron_reduce=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        self.S = num_sharesets

        self.h, self.w = h, w

        self.deploy = deploy

        assert in_channels == out_channels
        self.gp = GlobalPerceptron(input_channels=in_channels, internal_neurons=in_channels // globalperceptron_reduce)

        self.fc3 = nn.Conv2D(self.h * self.w * num_sharesets, self.h * self.w * num_sharesets, 1, 1, 0, bias_attr=deploy, groups=num_sharesets)
        if deploy:
            self.fc3_bn = Identity()
        else:
            self.fc3_bn = nn.BatchNorm2D(num_sharesets)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = conv_bn_relu(num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k//2, groups=num_sharesets, relu=False)
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def partition(self, x, h_parts, w_parts):
        x = x.reshape([-1, self.C, h_parts, self.h, w_parts, self.w])
        x = x.transpose([0, 2, 4, 1, 3, 5])
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape([-1, self.S * self.h * self.w, 1, 1])
        out = self.fc3(fc_inputs)
        out = out.reshape([-1, self.S, self.h, self.w])
        out = self.fc3_bn(out)
        out = out.reshape([-1, h_parts, w_parts, self.S, self.h, self.w])
        return out

    def forward(self, inputs):
        #   Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = inputs.shape
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        #   Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape([-1, self.S, self.h, self.w])
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape([-1, h_parts, w_parts, self.S, self.h, self.w])
            fc3_out += conv_out

        fc3_out = fc3_out.transpose([0, 3, 1, 4, 2, 5])  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape(fc_weight.shape) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        #   Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2D(self.S * self.h * self.w, self.S * self.h * self.w, 1, 1, 0, bias_attr=True, groups=self.S)
        self.fc3_bn = Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = paddle.eye(self.h * self.w).tile([1, self.S]).reshape([self.h * self.w, self.S, self.h, self.w])
        fc_k = F.conv2d(I, conv_kernel, padding=(conv_kernel.size(2)//2,conv_kernel.size(3)//2), groups=self.S)
        fc_k = fc_k.reshape([self.h * self.w, self.S * self.h * self.w]).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


#   The common FFN Block used in many Transformer and MLP models.
class FFNBlock(nn.Layer):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn_relu(in_channels, hidden_features, 1, 1, 0, relu=False)
        self.ffn_fc2 = conv_bn_relu(hidden_features, out_features, 1, 1, 0, relu=False)
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


class RepMLPNetUnit(nn.Layer):
    def __init__(self,
                 channels,
                 h,
                 w,
                 reparam_conv_k,
                 globalperceptron_reduce,
                 ffn_expand=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(in_channels=channels,
                                        out_channels=channels,
                                        h=h,
                                        w=w,
                                        reparam_conv_k=reparam_conv_k,
                                        globalperceptron_reduce=globalperceptron_reduce,
                                        num_sharesets=num_sharesets,
                                        deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2D(channels)
        self.prebn2 = nn.BatchNorm2D(channels)

    def forward(self, x):
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


class RepMLP(nn.Layer):
    """RepMLP Layer"""
    def __init__(self,
                 in_channels=3,
                 num_class=1000,
                 patch_size=(4, 4),
                 num_blocks=(2,2,6,2),
                 channels=(192,384,768,1536),
                 hs=(64,32,16,8),
                 ws=(64,32,16,8),
                 sharesets_nums=(4,8,16,32),
                 reparam_conv_k=(3,),
                 globalperceptron_reduce=4,
                 deploy=False):
        super().__init__()
        num_stages = len(num_blocks)
        assert num_stages == len(channels)
        assert num_stages == len(hs)
        assert num_stages == len(ws)
        assert num_stages == len(sharesets_nums)

        self.conv_embedding = conv_bn_relu(in_channels, channels[0], kernel_size=patch_size, stride=patch_size, padding=0)

        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = []
            for _ in range(num_blocks[stage_idx]):
                stage_blocks.append(RepMLPNetUnit(channels=channels[stage_idx],
                                                  h=hs[stage_idx],
                                                  w=ws[stage_idx],
                                                  reparam_conv_k=reparam_conv_k,
                                                  globalperceptron_reduce=globalperceptron_reduce,
                                                  ffn_expand=4,
                                                  num_sharesets=sharesets_nums[stage_idx],
                                                  deploy=deploy))
            stages.append(nn.LayerList(stage_blocks))
            if stage_idx < num_stages - 1:
                embeds.append(conv_bn_relu(in_channels=channels[stage_idx],
                                           out_channels=channels[stage_idx + 1],
                                           kernel_size=2,
                                           stride=2,
                                           padding=0))

        self.stages = nn.LayerList(stages)
        self.embeds = nn.LayerList(embeds)
        self.head_norm = nn.BatchNorm2D(channels[-1])
        self.head = nn.Linear(channels[-1], num_class)
        self.pool = nn.AdaptiveAvgPool2D(1)

    def forward(self, x):
        x = self.conv_embedding(x)
        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                x = embed(x)
        x = self.head_norm(x)
        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        x = self.head(x)
        return x

    def locality_injection(self):
        for m in self.sublayers():
            if hasattr(m, 'local_inject'):
                m.local_inject()


def build_repmlp(config):
    model = RepMLP(
        channels=config.MODEL.CHANNELS,
        hs=config.MODEL.HEIGHTS,
        ws=config.MODEL.WIDTHS,
        num_blocks=config.MODEL.NUM_BLOCKS,
        reparam_conv_k=config.MODEL.REPARAM_CONV_K,
        sharesets_nums=config.MODEL.SHARESETS_NUMS,
        deploy=config.MODEL.DEPLOY,
    )
    return model
