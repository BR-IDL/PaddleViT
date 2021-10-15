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


def TestRepMLP():
    # print('=== Test training_to_deploy for RepMLP ===')
    uniform_ = paddle.nn.initializer.Uniform(low=0, high=0.1, name=None)
    N = 1
    C = 8
    H = 14
    W = 14
    h = 7
    w = 7
    O = 8
    groups = 4

    x = paddle.randn([N, C, H, W])
    # print("input shape:", x.shape)
    repmlp = RepMLP(
        C,
        O,
        H=H,
        W=W,
        h=h,
        w=w,
        reparam_conv_k=(1, 3, 5),
        fc1_fc2_reduction=1,
        fc3_groups=groups,
        deploy=False,
    )
    repmlp.eval()

    for module in repmlp.sublayers():
        if isinstance(module, nn.BatchNorm2D) or isinstance(module, nn.BatchNorm1D):
            uniform_(module._mean)
            uniform_(module._variance)
            uniform_(module.weight)
            uniform_(module.bias)

    out = repmlp(x)
    repmlp.switch_to_deploy()
    deployout = repmlp(x)
    print("difference between the outputs of the training-time and converted RepMLP is")
    print(((deployout - out) ** 2).sum().numpy().item())


if __name__ == "__main__":
    TestRepMLP()
