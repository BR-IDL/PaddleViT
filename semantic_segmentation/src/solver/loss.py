"""Custom losses."""
import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .lovasz_losses import lovasz_softmax
from ..data.dataloader import datasets
from ..config import cfg

__all__ = ['get_segmentation_loss']

class TranslabLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(TranslabLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(TranslabLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(TranslabLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(TranslabLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(TranslabLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        loss = dict(loss=super(TranslabLoss, self).forward(*inputs))
        return loss

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, axis=1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index, axis=axis)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
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

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    def __init__(self, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).astype('float32')
        target_sub4 = F.interpolate(target, pred_sub4.shape[2:], mode='bilinear', align_corners=True).squeeze(1).astype('int64')
        target_sub8 = F.interpolate(target, pred_sub8.shape[2:], mode='bilinear', align_corners=True).squeeze(1).astype('int64')
        target_sub16 = F.interpolate(target, pred_sub16.shape[2:], mode='bilinear', align_corners=True).squeeze(1).astype('int64')
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Layer):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = paddle.to_tensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                       1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                       1.0865, 1.1529, 1.0507])
            self.criterion = paddle.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = paddle.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.shape
        target = target.reshape([-1])
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.astype('int64')
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, axis=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            # prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, paddle.arange(len(target), dtype='int64')]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.astype('int64')

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.reshape([n, h, w])

        return self.criterion(pred, target)


class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = cfg.MODEL.ENCNET.SE_LOSS
        self.se_weight = cfg.MODEL.ENCNET.SE_WEIGHT
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(paddle.nn.functional.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(paddle.nn.functional.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.shape[0]
        tvect = paddle.zeros([batch, nclass])
        for i in range(batch):
            hist = paddle.histogram(target[i].cpu().float(),
                                    bins=nclass, min=0,
                                    max=nclass-1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


class LovaszSoftmax(nn.Layer):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = lovasz_softmax(F.softmax(preds[0], axis=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], axis=1), target, ignore=self.ignore_index)
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
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class FocalLoss(nn.Layer):
    def __init__(self, alpha=0.5, gamma=2, weight=None, aux=True, aux_weight=0.2, ignore_index=-1,
                 size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = self._base_forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _base_forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().reshape([output.shape[0], output.shape[1], -1])
            output = output.transpose(1, 2)
            output = output.contiguous().reshape([-1, output.shape[2]]).squeeze()
        if target.dim() == 4:
            target = target.contiguous().reshape([target.shape[0], target.shape[1], -1])
            target = target.transpose(1, 2)
            target = target.contiguous().reshape([-1, target.shape[2]]).squeeze()
        elif target.dim() == 3:
            target = target.reshape([-1])
        else:
            target = target.reshape([-1, 1])

        logpt = self.ce_fn(output, target)
        pt = paddle.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class BinaryDiceLoss(nn.Layer):
    """
    Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: sum{x^p} + sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().reshape([predict.shape[0], -1])
        target = target.contiguous().reshape([target.shape[0], -1])
        valid_mask = valid_mask.contiguous().reshape([valid_mask.shape[0], -1])

        num = paddle.sum(paddle.multiply(predict, target) * valid_mask, axis=1) * 2 + self.smooth
        den = paddle.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, axis=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Layer):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, axis=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        valid_mask = (target != self.ignore_index).astype('int64')
        target_one_hot = F.one_hot(paddle.clip(target, 0))
        loss = self._base_forward(preds[0], target_one_hot, valid_mask)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'lovasz':
        logging.info('Use lovasz loss!')
        return LovaszSoftmax(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'focal':
        logging.info('Use focal loss!')
        return FocalLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'dice':
        logging.info('Use dice loss!')
        return DiceLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'binary_dice':
        logging.info('Use binary_dice loss!')
        return BinaryDiceLoss(**kwargs)

    model = model.lower()
    if model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'translab':
        return TranslabLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
