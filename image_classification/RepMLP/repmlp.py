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
Implement MLP Class for RepMLP
"""

import copy

import paddle
import paddle.nn.functional as F
from paddle import nn


def repeat_interleave(x, arg):
    """Use numpy to implement repeat operations"""
    return paddle.to_tensor(x.numpy().repeat(arg))


class Identity(nn.Layer):
    """Identity layer

    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def fuse_bn(conv_or_fc, bn):
    """Fusion of BN weights"""
    std = (bn._variance + bn._epsilon).sqrt()
    t = bn.weight / std
    if conv_or_fc.weight.ndim == 4:
        t = t.reshape([-1, 1, 1, 1])
    else:
        t = t.reshape([-1, 1])
    return conv_or_fc.weight * t, bn.bias - bn._mean * bn.weight / std


class RepMLP(nn.Layer):
    """RepMLP Layer

    The RepMLP consists of three parts: Global Perceptron, Partition Perceptron, Local Perceptron.
    When deploy is True, the training weight of Local Perceptron is integrated into the full connection
    layer of part of Partition Perceptron, In order to improve the ability of representation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        H,
        W,
        h,
        w,
        reparam_conv_k=None,
        fc1_fc2_reduction=1,
        fc3_groups=1,
        deploy=False,
    ):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        self.fc3_groups = fc3_groups

        self.H, self.W, self.h, self.w = H, W, h, w

        self.h_parts = self.H // self.h
        self.w_parts = self.W // self.w

        assert self.H % self.h == 0
        assert self.W % self.w == 0
        self.target_shape = (-1, self.O, self.H, self.W)

        self.deploy = deploy

        self.need_global_perceptron = (H != h) or (W != w)
        if self.need_global_perceptron:
            internal_neurons = int(
                self.C * self.h_parts * self.w_parts // fc1_fc2_reduction
            )
            self.fc1_fc2 = nn.Sequential()
            self.fc1_fc2.add_sublayer(
                "fc1", nn.Linear(self.C * self.h_parts * self.w_parts, internal_neurons)
            )
            self.fc1_fc2.add_sublayer("relu", nn.ReLU())
            self.fc1_fc2.add_sublayer(
                "fc2", nn.Linear(internal_neurons, self.C * self.h_parts * self.w_parts)
            )
            if deploy:
                self.avg = nn.AvgPool2D(kernel_size=(self.h, self.w))
            else:
                self.avg = nn.Sequential()
                self.avg.add_sublayer("avg", nn.AvgPool2D(kernel_size=(self.h, self.w)))
                self.avg.add_sublayer("bn", nn.BatchNorm2D(num_features=self.C))

        self.fc3 = nn.Conv2D(
            self.C * self.h * self.w,
            self.O * self.h * self.w,
            1,
            1,
            0,
            bias_attr=deploy,
            groups=fc3_groups,
        )
        self.fc3_bn = Identity() if deploy else nn.BatchNorm1D(self.O * self.h * self.w)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = nn.Sequential()
                conv_branch.add_sublayer(
                    "conv",
                    nn.Conv2D(
                        in_channels=self.C,
                        out_channels=self.O,
                        kernel_size=k,
                        padding=k // 2,
                        bias_attr=False,
                        groups=fc3_groups,
                    ),
                )
                conv_branch.add_sublayer("bn", nn.BatchNorm2D(self.O))
                self.__setattr__("repconv{}".format(k), conv_branch)

    def forward(self, inputs):

        if self.need_global_perceptron:
            v = self.avg(inputs)
            v = v.reshape([-1, self.C * self.h_parts * self.w_parts])
            v = self.fc1_fc2(v)
            v = v.reshape([-1, self.C, self.h_parts, 1, self.w_parts, 1])
            inputs = inputs.reshape(
                [-1, self.C, self.h_parts, self.h, self.w_parts, self.w]
            )
            inputs = inputs + v
        else:
            inputs = inputs.reshape(
                [-1, self.C, self.h_parts, self.h, self.w_parts, self.w]
            )

        # N, h_parts, w_parts, C, in_h, in_w
        partitions = inputs.transpose([0, 2, 4, 1, 3, 5])

        #   Feed partition map into Partition Perceptron
        fc3_inputs = partitions.reshape([-1, self.C * self.h * self.w, 1, 1])
        fc3_out = self.fc3(fc3_inputs)
        fc3_out = fc3_out.reshape([-1, self.O * self.h * self.w])
        fc3_out = self.fc3_bn(fc3_out)
        fc3_out = fc3_out.reshape(
            [-1, self.h_parts, self.w_parts, self.O, self.h, self.w]
        )

        #   Feed partition map into Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape([-1, self.C, self.h, self.w])
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__("repconv{}".format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(
                [-1, self.h_parts, self.w_parts, self.O, self.h, self.w]
            )
            fc3_out += conv_out

        # N, O, h_parts, out_h, w_parts, out_w
        fc3_out = fc3_out.transpose([0, 3, 1, 4, 2, 5])
        out = fc3_out.reshape([*self.target_shape])
        return out

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = (
            paddle.eye(self.C * self.h * self.w // self.fc3_groups)
            .tile(repeat_times=[1, self.fc3_groups])
            .reshape(
                [self.C * self.h * self.w // self.fc3_groups, self.C, self.h, self.w]
            )
        )
        fc_k = F.conv2d(
            I, conv_kernel, padding=conv_kernel.shape[2] // 2, groups=self.fc3_groups
        )
        fc_k = fc_k.reshape(
            [self.O * self.h * self.w // self.fc3_groups, self.C * self.h * self.w]
        ).t()
        fc_bias = repeat_interleave(conv_bias, self.h * self.w)
        return fc_k, fc_bias

    def get_equivalent_fc1_fc3_params(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)

        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__("repconv{}".format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__("repconv{}".format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias

            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape(fc_weight.shape) + fc_weight
            final_fc3_bias = rep_bias + fc_bias

        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias

        #   ------------------------------- remove BN after avg
        if self.need_global_perceptron:
            avgbn = self.avg.bn
            std = (avgbn._variance + avgbn._epsilon).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn._mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.weight.shape[0] // len(avgbias)
            replicated_avgbias = repeat_interleave(avgbias, replicate_times).reshape(
                [-1, 1]
            )
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            fc1_bias_new = fc1.bias + bias_diff
            fc1_weight_new = fc1.weight * repeat_interleave(
                scale, replicate_times
            ).reshape([1, -1])
        else:
            fc1_bias_new = None
            fc1_weight_new = None

        return fc1_weight_new, fc1_bias_new, final_fc3_weight, final_fc3_bias

    def switch_to_deploy(self):
        self.deploy = True
        (
            fc1_weight,
            fc1_bias,
            fc3_weight,
            fc3_bias,
        ) = self.get_equivalent_fc1_fc3_params()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__("repconv{}".format(k))
        #   Remove the BN after FC3
        self.__delattr__("fc3")
        self.__delattr__("fc3_bn")
        self.fc3 = nn.Conv2D(
            self.C * self.h * self.w,
            self.O * self.h * self.w,
            1,
            1,
            0,
            bias_attr=True,
            groups=self.fc3_groups,
        )
        self.fc3_bn = Identity()
        #   Remove the BN after AVG
        if self.need_global_perceptron:
            self.__delattr__("avg")
            self.avg = nn.AvgPool2D(kernel_size=(self.h, self.w))
        #   Set values
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.set_value(fc1_weight)
            self.fc1_fc2.fc1.bias.set_value(fc1_bias)
        self.fc3.weight.set_value(fc3_weight)
        self.fc3.bias.set_value(fc3_bias)


def repmlp_model_convert(model, save_path=None, do_copy=True):
    """reparameterizing model

    Args:
        model (nn.Layer): origin model
        save_path (str): save the model . Defaults to None.
        do_copy (bool): copy origin model. Defaults to True.

    Returns:
        nn.Layer: The reparameterized model
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.sublayers():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    if save_path is not None:
        paddle.save(model.state_dict(), save_path)
    return model


class ConvBN(nn.Layer):
    """Conv + BN"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        deploy=False,
        nonlinear=None,
    ):
        super().__init__()
        self.conv_in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = kernel_size
        self.conv_stride = stride
        self.conv_padding = padding
        self.conv_groups = groups

        if nonlinear is None:
            self.nonlinear = Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=True,
            )
        else:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=False,
            )
            self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, "bn"):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = fuse_bn(self.conv, self.bn)
        conv = nn.Conv2D(
            in_channels=self.conv_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.conv_kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            groups=self.conv_groups,
            bias_attr=True,
        )
        conv.weight.set_value(kernel)
        conv.bias.set_value(bias)
        self.__delattr__("conv")
        self.__delattr__("bn")
        self.conv = conv


class ConvBNReLU(ConvBN):
    """Conv + BN + ReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        deploy=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            deploy=deploy,
            nonlinear=nn.ReLU(),
        )


class RepMLPLightBlock(nn.Layer):
    """RepMLPLightBlock Layer

    The base module of the Light structure RepMLPResNet network
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        H,
        W,
        h,
        w,
        reparam_conv_k,
        fc1_fc2_reduction,
        fc3_groups,
        deploy=False,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = ConvBN(
                in_channels, out_channels, kernel_size=1, deploy=deploy
            )
        else:
            self.shortcut = Identity()
        self.light_conv1 = ConvBNReLU(
            in_channels, mid_channels, kernel_size=1, deploy=deploy
        )
        self.light_repmlp = RepMLP(
            in_channels=mid_channels,
            out_channels=mid_channels,
            H=H,
            W=W,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            fc1_fc2_reduction=fc1_fc2_reduction,
            fc3_groups=fc3_groups,
            deploy=deploy,
        )
        self.repmlp_nonlinear = nn.ReLU()
        self.light_conv3 = ConvBN(
            mid_channels, out_channels, kernel_size=1, deploy=deploy
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.light_conv1(x)
        out = self.light_repmlp(out)
        out = self.repmlp_nonlinear(out)
        out = self.light_conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


#   The input_ and output_channels of RepMLP are both mid_channels // r
class RepMLPBottleneckBlock(nn.Layer):
    """RepMLPBottleneckBlock Layer

    The base module of the bottleneck structure RepMLPResNet network
    """

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        r,
        H,
        W,
        h,
        w,
        reparam_conv_k,
        fc1_fc2_reduction,
        fc3_groups,
        deploy=False,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = ConvBN(
                in_channels, out_channels, kernel_size=1, deploy=deploy
            )
        else:
            self.shortcut = Identity()
        repmlp_channels = mid_channels // r
        self.btnk_conv1 = ConvBNReLU(
            in_channels, mid_channels, kernel_size=1, deploy=deploy
        )
        self.btnk_conv2 = ConvBNReLU(
            mid_channels, repmlp_channels, kernel_size=3, padding=1, deploy=deploy
        )
        self.btnk_repmlp = RepMLP(
            in_channels=repmlp_channels,
            out_channels=repmlp_channels,
            H=H,
            W=W,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            fc1_fc2_reduction=fc1_fc2_reduction,
            fc3_groups=fc3_groups,
            deploy=deploy,
        )
        self.repmlp_nonlinear = nn.ReLU()
        self.btnk_conv4 = ConvBNReLU(
            repmlp_channels, mid_channels, kernel_size=3, padding=1, deploy=deploy
        )
        self.btnk_conv5 = ConvBN(
            mid_channels, out_channels, kernel_size=1, deploy=deploy
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.btnk_conv1(x)
        out = self.btnk_conv2(out)
        out = self.btnk_repmlp(out)
        out = self.repmlp_nonlinear(out)
        out = self.btnk_conv4(out)
        out = self.btnk_conv5(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


#   Original block of ResNet-50


class BaseBlock(nn.Layer):
    """BaseBlock Layer

    Constitute the basic building blocks of a RepMLPResNet network
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(
                in_channels, out_channels, kernel_size=1, stride=stride, deploy=deploy
            )
        else:
            self.shortcut = Identity()
        self.conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1, deploy=deploy)
        self.conv2 = ConvBNReLU(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            deploy=deploy,
        )
        self.conv3 = ConvBN(mid_channels, out_channels, kernel_size=1, deploy=deploy)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RepMLPResNet(nn.Layer):
    """RepMLPResNet-50 Layer

    RepMLPResNet-50 has three structures:
    base: original ResNet-50
    light: RepMLP Light Block (55% faster, comparable accuracy)
    bottleneck: RepMLP Bottleneck Block (much higher accuracy, comparable speed)

    Args:
        block_type(str): "base", "light", "bottleneck"
    """

    def __init__(
        self,
        num_blocks,
        num_classes,
        block_type,
        img_H,
        img_W,
        h,
        w,
        reparam_conv_k,
        fc1_fc2_reduction,
        fc3_groups,
        deploy=False,
        # r=2 for stage2 and r=4 for stage3
        bottleneck_r=(2, 4),
    ):
        super().__init__()
        assert block_type in ["base", "light", "bottleneck"]
        self.block_type = block_type
        self.deploy = deploy

        self.img_H = img_H
        self.img_W = img_W
        self.h = h
        self.w = w
        self.reparam_conv_k = reparam_conv_k
        self.fc1_fc2_reduction = fc1_fc2_reduction
        self.fc3_groups = fc3_groups
        self.bottleneck_r = bottleneck_r

        self.in_channels = 64
        channels = [256, 512, 1024, 2048]

        self.stage0 = nn.Sequential(
            ConvBNReLU(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                deploy=deploy,
            ),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(
            channels[0], num_blocks[0], stride=1, total_downsample_ratio=4
        )
        self.stage2 = self._make_stage(
            channels[1], num_blocks[1], stride=2, total_downsample_ratio=8
        )
        self.stage3 = self._make_stage(
            channels[2], num_blocks[2], stride=2, total_downsample_ratio=16
        )
        self.stage4 = self._make_stage(
            channels[3], num_blocks[3], stride=2, total_downsample_ratio=32
        )
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)
        self.linear = nn.Linear(channels[3], num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.reshape([out.shape[0], -1])
        out = self.linear(out)
        return out

    def _make_stage(self, channels, num_blocks, stride, total_downsample_ratio):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for _, stride in enumerate(strides):
            # Only use RepMLP in stage2 and stage3, as described in the paper
            if (
                self.block_type == "base"
                or stride == 2
                or (total_downsample_ratio not in [8, 16])
            ):
                cur_block = BaseBlock(
                    in_channels=self.in_channels,
                    mid_channels=channels // 4,
                    out_channels=channels,
                    stride=stride,
                    deploy=self.deploy,
                )
            elif self.block_type == "light":
                cur_block = RepMLPLightBlock(
                    in_channels=self.in_channels,
                    mid_channels=channels // 8,
                    out_channels=channels,
                    H=self.img_H // total_downsample_ratio,
                    W=self.img_W // total_downsample_ratio,
                    h=self.h,
                    w=self.w,
                    reparam_conv_k=self.reparam_conv_k,
                    fc1_fc2_reduction=self.fc1_fc2_reduction,
                    fc3_groups=self.fc3_groups,
                    deploy=self.deploy,
                )
            elif self.block_type == "bottleneck":
                cur_block = RepMLPBottleneckBlock(
                    in_channels=self.in_channels,
                    mid_channels=channels // 4,
                    out_channels=channels,
                    r=self.bottleneck_r[0]
                    if total_downsample_ratio == 8
                    else self.bottleneck_r[1],
                    H=self.img_H // total_downsample_ratio,
                    W=self.img_W // total_downsample_ratio,
                    h=self.h,
                    w=self.w,
                    reparam_conv_k=self.reparam_conv_k,
                    fc1_fc2_reduction=self.fc1_fc2_reduction,
                    fc3_groups=self.fc3_groups,
                    deploy=self.deploy,
                )
            else:
                raise ValueError("Not supported.")

            blocks.append(cur_block)
            self.in_channels = channels

        return nn.Sequential(*blocks)


def build_repmlp(config):
    model = RepMLPResNet(
        num_blocks=config.MODEL.MIXER.NUM_BLOCKS,
        num_classes=config.MODEL.NUM_CLASSES,
        block_type=config.MODEL.MIXER.BLOCK_TYPE,
        img_H=config.MODEL.MIXER.IMG_H,
        img_W=config.MODEL.MIXER.IMG_W,
        h=config.MODEL.MIXER.H,
        w=config.MODEL.MIXER.W,
        reparam_conv_k=config.MODEL.MIXER.REPARAM_CONV_K,
        fc1_fc2_reduction=config.MODEL.MIXER.FC1_FC2_REDUCTION,
        fc3_groups=config.MODEL.MIXER.FC3_GROUPS,
        deploy=config.MODEL.MIXER.DEPLOY,
    )
    return model
