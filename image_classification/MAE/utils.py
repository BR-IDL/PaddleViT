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
from paddle.optimizer.lr import LRScheduler

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


def interpolate_pos_embed(model, state_dict):
    if 'position_embedding' in state_dict:
        pos_embed_w = state_dict['position_embedding']
        embed_dim = pos_embed_w.shape[-1]
        n_patches = model.patch_embedding.n_patches
        n_extra_tokens = model.position_embedding.shape[-2] - n_patches # seq_l - n_patches
        orig_size = int((pos_embed_w.shape[-2] - n_extra_tokens) ** 0.5)
        new_size = int(n_patches ** 0.5)
        if orig_size != new_size:
            extra_tokens = pos_embed_w[:, :n_extra_tokens]
            pos_tokens = pos_embed_w[:, n_extra_tokens:]
            pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size, embed_dim])
            pos_tokens = pos_tokens.transpose([0, 3, 1, 2])
            pos_tokens = paddle.nn.functional.interpolate(
                pos_token, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.transpose([0, 2, 3, 1])
            pos_tokens = pos_tokens.flatten(1, 2)
            new_pos_embed = paddle.concat([extra_tokens, pos_tokens], axis=1)
            state_dict['position_embedding'] = new_pos_embed


#TODO: check correctness
class LARS(paddle.optimizer.Optimizer):
    """LARS optmizer"""
    def __init__(self, params, learning_rate=0., weight_decay=0., momentum=0., trust_coefficient=0.001):
        super().__init__(params, learning_rate=learning_rate, weight_decay=weight_decay)

    @paddle.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue
                if p.ndim > 1:
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = paddle.norm(p)
                    update_norm = paddle.norm(dp)
                    one = paddle.ones_list(param_norm)
                    q = paddle.where(param_norm >0.,
                                     paddle.where(update_norm > 0,
                                                  (g['trust_coefficient'] * param_norm / update_norm),
                                                  one),
                                     one)
                    dp = dp.mul(q)
                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = paddle.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


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
