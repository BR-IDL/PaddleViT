# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
Implement Attention Layer
"""
import paddle
from paddle import nn

class Attention(nn.Layer):
    """Multi Head Attention
        Params Info:
            embed_dims: input token embed_dims
            num_head: the number of head is in multi head attention
            dropout_rate: the dropout rate of attention result
            attn_dropout_rate: the dropout rate of attention distribution
            qkv_bias: whether use the bias in qkv matrix
    """
    def __init__(self,
                 embed_dims,
                 num_head=1,
                 dropout_rate=0.,
                 attn_dropout_rate=0.,
                 qkv_bias=True):
        super(Attention, self).__init__(
                 name_scope="Attention")
        self.num_head = num_head
        self.head_dims = embed_dims // num_head
        self.scale = self.head_dims ** -0.5

        linear_weight_attr, linear_bias_attr = self._linear_init()

        self.qkv_proj = nn.Linear(in_features=embed_dims,
                                  out_features=3*self.num_head*self.head_dims,
                                  weight_attr=linear_weight_attr,
                                  bias_attr=linear_bias_attr if qkv_bias else qkv_bias)
        self.output = nn.Linear(in_features=self.num_head*self.head_dims,
                                out_features=embed_dims,
                                weight_attr=linear_weight_attr,
                                bias_attr=linear_bias_attr)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout= nn.Dropout(attn_dropout_rate)

    def _linear_init(self):
        weight_attr = nn.initializer.KaimingNormal()
        bias_attr = nn.initializer.Constant(value=0.0)
        return weight_attr, bias_attr

    def transfer_shape(self, q, k, v):
        B, M, _ = q.shape
        q = q.reshape(shape=[B, M, self.num_head, self.head_dims])
        q = q.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d
        k = k.reshape(shape=[B, M, self.num_head, self.head_dims])
        k = k.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d
        v = v.reshape(shape=[B, M, self.num_head, self.head_dims])
        v = v.transpose(perm=[0, 2, 1, 3]) # B, n_h, M, h_d

        return q, k, v

    def forward(self, inputs):
        B, M, D = inputs.shape
        assert D % self.num_head == 0, \
            "Erorr: Please make sure Token.D % "+\
            "num_head == 0(now:{0}).".format(D % self.num_head)

        qkv= self.qkv_proj(inputs)
        q, k, v = qkv.chunk(3, axis=-1)
        # B, n_h, M, h_d
        q, k, v = self.transfer_shape(q, k, v)

        attn = paddle.matmul(q, k, transpose_y=True) # B, n_h, M, M
        attn = attn * self.scale
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v) # B, n_h, M, h_d
        z = z.transpose(perm=[0, 2, 1, 3]) # B, M, n_h, h_d
        z = z.reshape(shape=[B, M, self.num_head*self.head_dims])
        z = self.output(z)
        z = self.attn_dropout(z)
        z = z + inputs

        return z