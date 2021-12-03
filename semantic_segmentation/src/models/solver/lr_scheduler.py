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

"""
Create Learning Rate Scheduler
"""

import math
import logging
from typing import List
from bisect import bisect_right
from paddle.optimizer.lr import LRScheduler
import paddle.optimizer.lr as lr_scheduler


_logger = logging.getLogger(__name__)


class WarmupCosineLR(LRScheduler):
    """WarmupCosineLR

    Apply Cosine learning rate with linear warmup

    Attributes:
        learning_rate: float, learning rate
        max_iters: int, can be total training steps
        t_mul: float, hyper for learning rate, default: 1.0
        lr_min: float, minimum learning rate, default: 0.0
        decay_rate: float, decay rate for cosine, 
                    if training steps greater than max_iters, default: 1.0
        warmup_steps: int, warmup steps, default: 0
        warmup_lr_init: float, initial warmup learning rate, default: 0.
        last_epoch: int, the index of last epoch. Can be set to restart training
                         default: -1, means initial learning rate
        **kwargs

    Examples:
        import paddle
        import matplotlib.pyplot as plt


        linear = paddle.nn.Linear(10, 10)
        scheduler = WarmupCosineLR(0.0005, 400, 1, 1e-05, 0.9, 40, 1e-06)
        sgd = paddle.optimizer.SGD(learning_rate=scheduler,
                                   parameters=linear.parameters())
        lr = []
        for epoch in range(400):
            lr.append(sgd.get_lr())
            scheduler.step()
        plt.plot(lr)
        plt.show()
    """
    def __init__(self,
                 learning_rate: float,
                 max_iters: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_steps=0,
                 warmup_lr_init=0.0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 last_epoch: int = -1,
                 verbose=False):
        assert max_iters > 0
        assert lr_min >= 0
        if max_iters == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                            "rate since max_iters = t_mul = eta_mul = 1.")
        self.max_iters = max_iters
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        if self.warmup_steps:
            self.warmup_iters = (learning_rate - warmup_lr_init) / self.warmup_steps
        else:
            self.warmup_iters = 1
        super(WarmupCosineLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr = self.warmup_lr_init + self.last_epoch * self.warmup_iters
        else:
            if self.warmup_prefix:
                self.last_epoch = self.last_epoch - self.warmup_steps
            if self.t_mul != 1:
                i = math.floor(math.log(1 - self.last_epoch / self.max_iters * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.max_iters
                t_curr = self.last_epoch - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.max_iters
            else:
                i = self.last_epoch // self.max_iters
                t_i = self.max_iters
                t_curr = self.last_epoch - (self.max_iters * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = self.base_lr * gamma
            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lr = lr_min + 0.5 * (lr_max_values - lr_min) * (1 + math.cos(math.pi * t_curr / t_i))
            else:
                lr = self.lr_min
        return lr


class WarmupPolyLR(LRScheduler):
    """WarmupPolyLR

    Apply PolynomialDecay learning rate with linear warmup

    Attributes:
        learning_rate: float, learning rate
        warmup_lr_init: float, initial lrarning rate for warmup, default: 0.0
        max_iters: int, total training steps
        power: float, Power of polynomial, default: 0.9.
        lr_min: float, minimum learning rate, default: 0.0
        warmup_steps: int, warmup steps, default: 0
        last_epoch: int, the index of last epoch. Can be set to restart training.
                         default: -1, means initial learning rate

    Examples:
        import paddle.nn as nn
        from paddle.optimizer import Adam
        import matplotlib.pyplot as plt


        scheduler = WarmupPolyLR(1e-4,
                                 max_iters=200,
                                 power=0.9,
                                 warmup_steps=30)
        opt = Adam(parameters=nn.Linear(10, 10).parameters(), learning_rate=scheduler)
        lr = []
        for epoch in range(0, 1000):
            lr.append(opt.get_lr())
            scheduler.step(epoch)
        plt.plot(lr)
        plt.show()
    """
    def __init__(self,
                 learning_rate,
                 warmup_lr_init=0,
                 max_iters=0,
                 power=0.9,
                 warmup_steps=5,
                 lr_min=0.0,
                 last_epoch=-1,
                 verbose=False):
        self.base_lr = float(learning_rate)
        self.warmup_lr_init = warmup_lr_init
        self.max_iters = max_iters
        self.power = power
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        assert learning_rate > lr_min, _logger.error('learning_rate must >= lr_min:{}'.format(lr_min))
        super(WarmupPolyLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        N = self.max_iters - self.warmup_steps
        T = self.last_epoch - self.warmup_steps
        if self.last_epoch < self.warmup_steps:
            warmup_factor = float(self.last_epoch) / self.warmup_steps
            if self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * warmup_factor <= self.lr_min:
                return self.lr_min
            return self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * warmup_factor
        factor = pow(1 - T / N, self.power)
        if isinstance(self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * factor, complex):
            return self.lr_min
        if self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * factor <= self.lr_min:
            return self.lr_min
        return self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * factor


class WarmupMultiStepLR(LRScheduler):
    """WarmupMultiStepLR

    Apply MultiStep learning rate with linear warmup

    Attributes:
        learning_rate: float, learning rate
        milestones: (tuple|list),  List or tuple of each boundaries. Must be increasing
        gamma: float, the Ratio that the learning rate will be reduced, default: 0.1
        warmup_steps: int, warmup steps, default: 0
        last_epoch: int, the index of last epoch. Can be set to restart training.
                         default: -1, means initial learning rate

    Examples:
        import paddle.nn as nn
        from paddle.optimizer import Adam
        import matplotlib.pyplot as plt


        scheduler = WarmupMultiStepLR(0.001,
                                      milestones=[50, 150, 200],
                                      gamma=0.1,
                                      warmup_steps=50)
        opt = Adam(parameters=nn.Linear(10, 10).parameters(), learning_rate=scheduler)
        lr = []
        for epoch in range(0, 500):
            lr.append(opt.get_lr())
            scheduler.step(epoch)
        plt.plot(lr)
        plt.show()
    """
    def __init__(self,
                 learning_rate: float,
                 milestones: List[int],
                 gamma: float = 0.1,
                 warmup_steps: int = 1000,
                 last_epoch: int = -1,
                 verbose=False):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.base_lr = float(learning_rate)
        assert self.warmup_steps <= milestones[0], _logger.error('warmup steps must >= milestones[0]')
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_steps:
            warmup_factor = float(self.last_epoch) / self.warmup_steps
            return self.base_lr * warmup_factor
        return self.base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)


def get_scheduler(config):
    if config.TRAIN.LR_SCHEDULER.NAME == 'PolynomialDecay':
        scheduler = lr_scheduler.PolynomialDecay(learning_rate=config.TRAIN.BASE_LR,
                                                 decay_steps=config.TRAIN.ITERS,
                                                 end_lr=config.TRAIN.END_LR,
                                                 power=config.TRAIN.POWER)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'WarmupCosineLR':
        scheduler = WarmupCosineLR(learning_rate=config.TRAIN.BASE_LR,
                                   max_iters=config.TRAIN.ITERS,
                                   warmup_steps=config.TRAIN.LR_SCHEDULER.WARM_UP_STEPS,
                                   warmup_lr_init=config.TRAIN.LR_SCHEDULER.WARM_UP_LR_INIT,
                                   lr_min=config.TRAIN.END_LR)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'WarmupPolyLR':
        scheduler = WarmupPolyLR(learning_rate=config.TRAIN.BASE_LR,
                                 max_iters=config.TRAIN.ITERS,
                                 power=config.TRAIN.LR_SCHEDULER.POWER,
                                 warmup_lr_init=config.TRAIN.LR_SCHEDULER.WARM_UP_LR_INIT,
                                 warmup_steps=config.TRAIN.LR_SCHEDULER.WARM_UP_STEPS,
                                 lr_min=config.TRAIN.END_LR)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'WarmupMultiStepLR':
        scheduler = WarmupMultiStepLR(learning_rate=config.TRAIN.BASE_LR,
                                      milestones=config.TRAIN.LR_SCHEDULER.MILESTONES,
                                      gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
                                      warmup_steps=config.TRAIN.LR_SCHEDULER.WARM_UP_STEPS)
    return scheduler
