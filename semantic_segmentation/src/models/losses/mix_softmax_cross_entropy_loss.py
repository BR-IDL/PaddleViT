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

"""MixSoftmaxCrossEntropyLoss Implement
"""
import paddle.nn as nn


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    """MixSoftmaxCrossEntropyLoss
    """
    def __init__(self, config):
        self.ignore_index = config.TRAIN.IGNORE_INDEX
        self.aux = config.MODEL.AUX.LOSS
        self.aux_weight = config.MODEL.AUX.AUX_WEIGHT
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=self.ignore_index, axis=1)

    def _aux_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        if len(preds) > 1:
            return self._multiple_forward(*inputs)
        return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)
