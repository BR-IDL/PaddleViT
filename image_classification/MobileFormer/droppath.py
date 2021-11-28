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
Implement Multi-Branch Dropout Layer
"""
import paddle
from paddle import nn

class DropPath(nn.Layer):
    """Multi-branch dropout layer -- Along the axis of Batch
        Params Info:
            p: droppath rate
    """
    def __init__(self,
                 p=0.):
        super(DropPath, self).__init__(
                 name_scope="DropPath")
        self.p = p

    def forward(self, inputs):
        if self.p > 0. and self.training:
            keep_p = 1 - self.p
            keep_p = paddle.to_tensor([keep_p])
            # B, 1, 1....
            shape = [inputs.shape[0]] + [1] * (inputs.ndim-1)
            random_dr = keep_p + paddle.rand(shape=shape, dtype='float32')
            random_sample = random_dr.floor() # floor to int--B
            output = inputs.divide(keep_p) * random_sample
            return output

        return inputs
