# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

""" Implement Loss functions """
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TokenLabelGTCrossEntropy(nn.Layer):
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_activate=True,
                 smoothing=0.1,
                 classes=1000):
        super().__init__()
        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.smoothing = smoothing
        self.mixup_activate = mixup_activate
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):
        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb
        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
#TODO: fix bugs
            target_aux = target.expand([1, N]).reshape((B*N, C))
        else:
            ground_truth = target[:, :, 0]
            target_cls = target[:, :, 1]
            ratio = (0.9 - 0.4 * (ground_truth.max(-1)[1] == target_cls.max(-1)[1])).unsqueeze(-1)
            target_cls = target_cls * ratio + ground_truth * (1 - ratio)
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose([0, 2, 1]).reshape((-1, C))
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape((-1, C))

        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)

        return self.cls_weigth * loss_cls + self.dense_weight * loss_aux



class TokenLabelCrossEntropy(nn.Layer):
    def __init__(self,
                 dense_weight=1.0,
                 cls_weight=1.0,
                 mixup_activate=True,
                 classes=1000):
        super().__init__()
        self.CE = SoftTargetCrossEntropy()

        self.dense_weight = dense_weight
        self.mixup_activate = mixup_activate
        self.classes = classes
        self.cls_weight = cls_weight
        assert dense_weight + cls_weight > 0

    def forward(self, x, target):
        output, aux_output, bb = x
        bbx1, bby1, bbx2, bby2 = bb
        B, N, C = aux_output.shape
        if len(target.shape) == 2:
            target_cls = target
#TODO: fix bugs
            target_aux = target.expand([1, N]).reshape((B*N, C))
        else:
            target_cls = target[:, :, 1]
            target_aux = target[:, :, 2:]
            target_aux = target_aux.transpose([0, 2, 1]).reshape((-1, C))
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / N)
        if lam < 1:
            target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)

        aux_output = aux_output.reshape((-1, C))

        loss_cls = self.CE(output, target_cls)
        loss_aux = self.CE(aux_output, target_aux)

        return self.cls_weigth * loss_cls + self.dense_weight * loss_aux


class TokenLabelSoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N == N_rep:
# TODO:
            target = target.repeat(N_rep // N, 1)
        if len(target.shape) == 3 and target.shape[-1] == 2:
            target = target[:, :, 1]
        loss = paddle.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class LabelSmoothingCrossEntropyLoss(nn.Layer):
    """ cross entropy loss for label smoothing
    Args:
        smoothing: float, smoothing rate
        x: tensor, predictions (before softmax) with shape [N, num_classes]
        target: tensor, target label with shape [N]
    Return:
        loss: float, cross entropy loss value
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1 - smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x) # [N, num_classes]
        # target_index is used to get prob for each of the N samples
        target_index = paddle.zeros([x.shape[0], 2], dtype='int64') # [N, 2]
        target_index[:, 0] = paddle.arange(x.shape[0])
        target_index[:, 1] = target

        nll_loss = -log_probs.gather_nd(index=target_index) # index: [N]
        smooth_loss = -log_probs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropyLoss(nn.Layer):
    """ cross entropy loss for soft target
    Args:
        x: tensor, predictions (before softmax) with shape [N, num_classes]
        target: tensor, soft target with shape [N, num_classes]
    Returns:
        loss: float, the mean loss value
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        return loss.mean()


class DistillationLoss(nn.Layer):
    """Distillation loss function
    This layer includes the orginal loss (criterion) and a extra 
    distillation loss (criterion), which computes the loss with 
    different type options, between current model and 
    a teacher model as its supervision.

    Args:
        base_criterion: nn.Layer, the original criterion
        teacher_model: nn.Layer, the teacher model as supervision
        distillation_type: str, one of ['none', 'soft', 'hard']
        alpha: float, ratio of base loss (* (1-alpha)) 
               and distillation loss( * alpha)
        tao: float, temperature in distillation
    """
    def __init__(self,
                 base_criterion,
                 teacher_model,
                 distillation_type,
                 alpha,
                 tau):
        super().__init__()
        assert distillation_type in ['none', 'soft', 'hard']
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, targets):
        """
        Args:
            inputs: tensor, the orginal model inputs
            outputs: tensor, the outputs of the model
            outputds_kd: tensor, the distillation outputs of the model,
                         this is usually obtained by a separate branch
                         in the last layer of the model
            targets: tensor, the labels for the base criterion
        """
        outputs, outputs_kd = outputs[0], outputs[1]
        base_loss = self.base_criterion(outputs, targets)
        if self.type == 'none':
            return base_loss

        with paddle.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.type == 'soft':
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / self.tau, axis=1),
                F.log_softmax(teacher_outputs / self.tau, axis=1),
                reduction='sum') * (self.tau * self.tau) / outputs_kd.numel()
        elif self.type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(axis=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


