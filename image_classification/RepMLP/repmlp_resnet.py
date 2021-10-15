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

import paddle
import paddle.nn.functional as F
from paddle import nn

from repmlp import Identity, RepMLP, fuse_bn, repmlp_model_convert


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
            in_channels=self.conv._in_channels,
            out_channels=self.conv._out_channels,
            kernel_size=self.conv._kernel_size,
            stride=self.conv._stride,
            padding=self.conv._padding,
            groups=self.conv._groups,
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


def build_repmlp_resnet(config):
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


def TestConvBN():
    print("=== Test training_to_deploy for ConvBN ===")
    x = paddle.randn([1, 5, 22, 22])
    print("input shape:", x.shape)
    m = ConvBN(5, 10, 3)
    m.eval()

    out = m(x)
    m.switch_to_deploy()
    deployout = m(x)
    print("difference between the outputs of the training-time and converted ConvBN")
    print(((deployout - out) ** 2).sum().numpy().item())


def TestModel():
    print("=== Test training_to_deploy for RepMLP_ResNet ===")

    x = paddle.randn([1, 3, 224, 224])
    print("input shape:", x.shape)

    model = RepMLPResNet(
        num_blocks=[3, 4, 6, 3],
        num_classes=1000,
        block_type="light",
        img_H=224,
        img_W=224,
        h=7,
        w=7,
        reparam_conv_k=(1, 3, 5),
        fc1_fc2_reduction=1,
        fc3_groups=4,
        deploy=False,
    )
    model.eval()

    out = model(x)
    deploy_model = repmlp_model_convert(model)
    deployout = deploy_model(x)
    print(
        "difference between the outputs of the training-time and converted RepMLP_ResNet"
    )
    print(((deployout - out) ** 2).sum().numpy().item())
    print("Done!")


if __name__ == "__main__":
    TestConvBN()
    TestModel()
