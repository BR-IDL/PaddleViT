import paddle.nn as nn


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.4, ignore_index=-1, axis=1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index, axis=axis)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target) # [8, 12, 512, 512] [8, 512, 512]
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

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            # 这里
            return self._aux_forward(*inputs)
        elif len(preds) > 1:
            return self._multiple_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)