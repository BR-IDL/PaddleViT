# this code heavily reference: detectron2
from __future__ import division
import math
from paddle.optimizer.lr import LRScheduler

from typing import List
from bisect import bisect_right
from segmentron.config import cfg

__all__ = ['get_scheduler']


class WarmupPolyLR(LRScheduler):
    def __init__(self, learning_rate, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))
        self.base_lr = float(learning_rate)
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(learning_rate, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return self.target_lr + (self.base_lr - self.target_lr) * warmup_factor
        factor = pow(1 - T / N, self.power)
        return self.target_lr + (self.base_lr - self.target_lr) * factor


class WarmupMultiStepLR(LRScheduler):
    def __init__(
        self,
        learning_rate: float,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.base_lr = float(learning_rate)
        super().__init__(learning_rate, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return self.base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)


class WarmupCosineLR(LRScheduler):
    def __init__(
        self,
        learning_rate: float,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.base_lr = float(learning_rate)
        super().__init__(learning_rate, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return self.base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def get_scheduler(learning_rate, max_iters, iters_per_epoch):
    mode = cfg.SOLVER.LR_SCHEDULER.lower()
    warm_up_iters = iters_per_epoch * cfg.SOLVER.WARMUP.EPOCHS
    if mode == 'poly':
        return WarmupPolyLR(learning_rate, max_iters=max_iters, power=cfg.SOLVER.POLY.POWER,
                            warmup_factor=cfg.SOLVER.WARMUP.FACTOR, warmup_iters=warm_up_iters,
                            warmup_method=cfg.SOLVER.WARMUP.METHOD)
    elif mode == 'cosine':
        return WarmupCosineLR(learning_rate, max_iters=max_iters, warmup_factor=cfg.SOLVER.WARMUP.FACTOR,
                              warmup_iters=warm_up_iters, warmup_method=cfg.SOLVER.WARMUP.METHOD)
    elif mode == 'step':
        milestones = [x * iters_per_epoch for x in cfg.SOLVER.STEP.DECAY_EPOCH]
        return WarmupMultiStepLR(learning_rate, milestones=milestones, gamma=cfg.SOLVER.STEP.GAMMA,
                                 warmup_factor=cfg.SOLVER.WARMUP.FACTOR, warmup_iters=warm_up_iters,
                                 warmup_method=cfg.SOLVER.WARMUP.METHOD)
    else:
        raise ValueError("not support lr scheduler method!")

