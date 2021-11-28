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
Implement MLP And DYReLU Layer
"""
import paddle
from paddle import nn

class MLP(nn.Layer):
    """Multi Layer Perceptron
        Params Info:
            in_features: input token feature size
            out_features: output token feature size
            mlp_ratio: the scale of hidden feature size
            mlp_dropout_rate: the dropout rate of mlp layer output
    """
    def __init__(self,
                 in_features,
                 out_features=None,
                 mlp_ratio=2,
                 mlp_dropout_rate=0.,
                 act=nn.GELU):
        super(MLP, self).__init__(name_scope="MLP")
        self.out_features = in_features if out_features is None else \
                            out_features
        linear_weight_attr, linear_bias_attr = self._linear_init()

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=int(mlp_ratio*in_features),
                             weight_attr=linear_weight_attr,
                             bias_attr=linear_bias_attr)
        self.fc2 = nn.Linear(in_features=int(mlp_ratio*in_features),
                             out_features=self.out_features,
                             weight_attr=linear_weight_attr,
                             bias_attr=linear_bias_attr)

        self.act = act()
        self.dropout = nn.Dropout(mlp_dropout_rate)

    def _linear_init(self):
        weight_attr = nn.initializer.KaimingNormal()
        bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DyReLU(nn.Layer):
    """Dynamic ReLU activation function -- use one MLP
        Params Info:
            in_channels: input feature map channels
            embed_dims: input token embed_dims
            k: the number of parameters is in Dynamic ReLU
            coefs: the init value of coefficient parameters
            consts: the init value of constant parameters
            reduce: the mlp hidden scale,
                    means 1/reduce = mlp_ratio
    """
    def __init__(self,
                 in_channels,
                 embed_dims,
                 k=2, # a_1, a_2 coef, b_1, b_2 bias
                 coefs=[1.0, 0.5], # coef init value
                 consts=[1.0, 0.0], # const init value
                 reduce=4):
        super(DyReLU, self).__init__(
                 name_scope="DyReLU")
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.k = k

        self.mid_channels = 2*k*in_channels

        # 4 values
        # a_k = alpha_k + coef_k*x, 2
        # b_k = belta_k + coef_k*x, 2
        self.coef = paddle.to_tensor([coefs[0]]*k + [coefs[1]]*k)
        self.const = paddle.to_tensor([consts[0]] + [consts[1]]*(2*k-1))

        self.project = nn.Sequential(
            MLP(in_features=embed_dims,
                out_features=self.mid_channels,
                mlp_ratio=1/reduce,
                act=nn.ReLU),
            nn.BatchNorm(self.mid_channels)
        )

    def forward(self, feature_map, tokens):
        B, M, D = tokens.shape
        dy_params = self.project(tokens[:, 0]) # B, mid_channels
        # B, IN_CHANNELS, 2*k
        dy_params = dy_params.reshape(shape=[B, self.in_channels, 2*self.k])

        # B, IN_CHANNELS, 2*k -- a_1, a_2, b_1, b_2
        dy_init_params = dy_params * self.coef + self.const
        f = feature_map.transpose(perm=[2, 3, 0, 1]).unsqueeze(axis=-1) # H, W, B, C, 1

        # output shape: H, W, B, C, k
        output = f * dy_init_params[:, :, :self.k] + dy_init_params[:, :, self.k:]
        output = paddle.max(output, axis=-1) # H, W, B, C
        output = output.transpose(perm=[2, 3, 0, 1]) # B, C, H, W

        return output
