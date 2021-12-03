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
        elif len(preds) > 1:
            return self._multiple_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)
