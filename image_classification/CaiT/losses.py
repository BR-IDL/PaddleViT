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


class LabelSmoothingCrossEntropyLoss(nn.Layer):
    """ cross entropy loss for label smoothing
    Args:
        smoothing: float, label smoothing rate
        x: tensor, predictions (default is before softmax) with shape [N, num_classes] as default
        target: tensor, target label with shape [N] as default
        weight: tensor, optional, a manual rescaling weight given to each class        
        reduction: str, optional, indicate how to average the loss by batch_size,
                   default is ``'mean'``, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``
        axis: int, optional, the index of dimension to perform softmax calculations,
                   default is ``-1``, if `axis` is not -1 -> the shape of x and target may not be default
        use_softmax: bool, optional, if `use_softmax` is ``False``, ``x`` should be after softmax,
                     default is ``True``, the candicates are ``True`` | ``False``
        name: str, optional, the name of the operator, default is ``None``,
              for more information, please refer to :ref:`api_guide_Name`.
    Return:
        loss: float, cross entropy loss value
    """
    def __init__(self,
                 smoothing=0.1,
                 weight=None,                 
                 reduction='mean',                 
                 axis=-1,
                 use_softmax=True,
                 name=None):
        super().__init__()
        assert 0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction        
        self.axis = axis
        self.use_softmax = use_softmax
        self.name = name

    def forward(self, x, target):
        target = paddle.nn.functional.one_hot(target, num_classes=x.shape[1])
        target = paddle.nn.functional.label_smooth(target, epsilon=self.smoothing)        
        loss = paddle.nn.functional.cross_entropy(
            x,
            target,            
            weight=self.weight,            
            reduction=self.reduction,
            soft_label=True,
            axis=self.axis,
            use_softmax=self.use_softmax,
            name=self.name)
        return loss


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


