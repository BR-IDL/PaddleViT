from .cross_entropy_loss import CrossEntropyLoss
from .mix_softmax_cross_entropy_loss import MixSoftmaxCrossEntropyLoss
from .multi_cross_entropy_loss import MultiCrossEntropyLoss


def get_loss_function(config):
    if config.TRAIN.LOSS == 'CrossEntropyLoss':
        return CrossEntropyLoss()
    if config.TRAIN.LOSS == 'MixSoftmaxCrossEntropyLoss':
        return MixSoftmaxCrossEntropyLoss(config)
    if config.TRAIN.LOSS == 'MultiCrossEntropyLoss':
        return MultiCrossEntropyLoss(config)
