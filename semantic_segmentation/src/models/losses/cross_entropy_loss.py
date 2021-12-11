#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

import paddle
from paddle import nn
import paddle.nn.functional as F


class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (ndarray, optional): A manual rescaling weight given to each 
        class. Its length must be equal to the number of classes. Default ``None``.
        ignore_index (int64, optional): The ignored class. 
    """

    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = paddle.to_tensor(weight, dtype='float32')
        self.weight = weight
        self.ignore_index = ignore_index
        self.eps = 1e-8

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor. shape: (B,C,H,W)
            label (Tensor): Label tensor, the data type is int64. shape: (B,H,W)
        """
        if self.weight is not None and logit.shape[1] != len(self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[1]))

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        if self.weight is None:
            loss = F.cross_entropy(
                logit, label, ignore_index=self.ignore_index, reduction='none')
        else:
            label_one_hot = F.one_hot(label, logit.shape[-1])
            loss = F.cross_entropy(
                logit,
                label_one_hot * self.weight,
                soft_label=True,
                ignore_index=self.ignore_index,
                reduction='none')
            loss = loss.squeeze(-1)

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights
        label.stop_gradient = True
        mask.stop_gradient = True
        avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.eps)
        return avg_loss
