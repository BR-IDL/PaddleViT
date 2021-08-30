import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# TODO: add related classes and methods for segmentations
def dice_loss(inputs, targets, num_boxes):
    inputs = F.sigmoid(inputs)
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominaror = inputs.sum(-1) + target.sum(-1)
    loss = 1 - (numerator + 1) / (denominator +1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=.25, gamma=2.):
    prob = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
