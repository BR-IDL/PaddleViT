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

"""utils for ViT

Contains AverageMeter for monitoring, get_exclude_from_decay_fn for training
and WarmupCosineScheduler for training

"""

import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.optimizer.lr import LRScheduler


def cancle_gradient_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if 'last_layer' in n:
            p.stop_gradient = True


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D, nn.SyncBatchNorm)
    for name, layer in model.named_sublayers():
        if isinstance(layer, bn_types):
            return True
    return False


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        # do not regularize biases and norm params
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     num_iters_per_epoch,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * num_iters_per_epoch
    if warmup_epochs > 0:
        # linear schedule for warmup epochs 
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * num_iters_per_epoch  - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * num_iters_per_epoch
    return schedule


class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



def get_exclude_from_weight_decay_fn(exclude_list=[]):
    """ Set params with no weight decay during the training

    For certain params, e.g., positional encoding in ViT, weight decay
    may not needed during the learning, this method is used to find
    these params.

    Args:
        exclude_list: a list of params names which need to exclude
                      from weight decay.
    Returns:
        exclude_from_weight_decay_fn: a function returns True if param
                                      will be excluded from weight decay
    """
    if len(exclude_list) == 0:
        exclude_from_weight_decay_fn = None
    else:
        def exclude_fn(param):
            for name in exclude_list:
                if param.endswith(name):
                    return False
            return True
        exclude_from_weight_decay_fn = exclude_fn
    return exclude_from_weight_decay_fn


class WarmupCosineScheduler(LRScheduler):
    """Warmup Cosine Scheduler

    First apply linear warmup, then apply cosine decay schedule.
    Linearly increase learning rate from "warmup_start_lr" to "start_lr" over "warmup_epochs"
    Cosinely decrease learning rate from "start_lr" to "end_lr" over remaining
    "total_epochs - warmup_epochs"

    Attributes:
        learning_rate: the starting learning rate (without warmup), not used here!
        warmup_start_lr: warmup starting learning rate
        start_lr: the starting learning rate (without warmup)
        end_lr: the ending learning rate after whole loop
        warmup_epochs: # of epochs for warmup
        total_epochs: # of total epochs (include warmup)
    """
    def __init__(self,
                 learning_rate,
                 warmup_start_lr,
                 start_lr,
                 end_lr,
                 warmup_epochs,
                 total_epochs,
                 cycles=0.5,
                 last_epoch=-1,
                 verbose=False):
        """init WarmupCosineScheduler """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.cycles = cycles
        super(WarmupCosineScheduler, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        """ return lr value """
        if self.last_epoch < self.warmup_epochs:
            val = (self.start_lr - self.warmup_start_lr) * float(
                self.last_epoch)/float(self.warmup_epochs) + self.warmup_start_lr
            return val

        progress = float(self.last_epoch - self.warmup_epochs) / float(
            max(1, self.total_epochs - self.warmup_epochs))
        val = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
        val = max(0.0, val * (self.start_lr - self.end_lr) + self.end_lr)
        return val
