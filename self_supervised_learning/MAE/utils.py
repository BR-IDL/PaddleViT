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

import logging
import sys
import os
import paddle
from paddle.optimizer.lr import LRScheduler


def get_logger(file_path):
    """Set logging file and format, logs are written in 2 loggers, one local_logger records
       the information on its own gpu/process, one master_logger records the overall/average
       information over all gpus/processes.
    Args:
        file_path: str, folder path of the logger files to write
    Return:
        local_logger: python logger for each process
        master_logger: python logger for overall processes (on node 0)
    """
    local_rank = paddle.distributed.get_rank()
    filename = os.path.join(file_path, 'log_all.txt')
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")

    # local_logger for each process/GPU
    local_logger = logging.getLogger(f'local_{local_rank}')
    filename = os.path.join(file_path, f'log_{local_rank}.txt')
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(log_format))
    local_logger.addHandler(fh)

    # master_logger records avg performance and general message
    if local_rank == 0:
        master_logger = logging.getLogger('master')
        # log.txt
        filename = os.path.join(file_path, 'log.txt')
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(fh)
        # consol (stdout)
        sh_1 = logging.StreamHandler(sys.stdout)
        sh_1.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(sh_1)
        # consol (stderr)
        sh_2 = logging.StreamHandler(sys.stderr)
        sh_2.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(sh_2)
    else:
        master_logger = None
    return local_logger, master_logger


def write_log(local_logger, master_logger, msg_local, msg_master=None, level='info'):
    """Write messages in loggers
    Args:
        local_logger: python logger, logs information on single gpu
        master_logger: python logger, logs information over all gpus
        msg_local: str, message to log on local_logger
        msg_master: str, message to log on master_logger, if None, use msg_local, default: None
        level: str, log level, in ['info', 'warning', 'fatal'], default: 'info'
    """
    # write log to local logger
    if local_logger:
        if level == 'info':
            local_logger.info(msg_local)
        elif level == 'warning':
            local_logger.warning(msg_local)
        elif level == 'fatal':
            local_logger.fatal(msg_local)
        else:
            raise ValueError("level must in ['info', 'warning', 'fatal']")
    # write log to master logger on node 0
    if master_logger and paddle.distributed.get_rank() == 0:
        if msg_master is None:
            msg_master = msg_local
        if level == 'info':
            master_logger.info("MASTER_LOG " + msg_master)
        elif level == 'warning':
            master_logger.warning("MASTER_LOG " + msg_master)
        elif level == 'fatal':
            master_logger.fatal("MASTER_LOG " + msg_master)
        else:
            raise ValueError("level must in ['info', 'warning', 'fatal']")


def all_reduce_mean(x):
    """perform all_reduce on Tensor for gathering results from multi-gpus"""
    world_size = paddle.distributed.get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        paddle.distributed.all_reduce(x_reduce)
        x_reduce = x_reduce / world_size
        return x_reduce.item()
    return x


def get_params_groups(model, weight_decay=0.01):
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
    return [{'params': regularized, 'weight_decay': weight_decay}, {'params': not_regularized, 'weight_decay': 0.}]


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


def adjust_learning_rate(optimizer,
                         base_lr,
                         min_lr,
                         cur_epoch,
                         warmup_epochs,
                         total_epochs):
    if cur_epoch < warmup_epochs:
        lr = base_lr * cur_epoch / warmup_epochs
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1. + math.cos(math.pi * (cur_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    optimizer.set_lr(lr)
    return lr


def interpolate_pos_embed(model, state_dict, key_name='encoder_position_embedding'):
    if key_name in state_dict:
        pos_embed_w = state_dict[key_name]
        embed_dim = pos_embed_w.shape[-1]
        n_patches = model.patch_embedding.n_patches
        n_extra_tokens = getattr(model, key_name).shape[-2] - n_patches
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
            state_dict[key_name] = new_pos_embed


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


def skip_weight_decay_fn(model, skip_list=[], filter_bias_and_bn=True):
    """ Set params with no weight decay during the training

    For certain params, e.g., positional encoding in ViT, weight decay
    may not needed during the learning, this method is used to find
    these params.

    Args:
		model: nn.Layer, model
        skip_list: list, a list of params names which need to exclude
                      from weight decay, default: []
		filter_bias_and_bn: bool, set True to exclude bias and bn in model, default: True
    Returns:
        exclude_from_weight_decay_fn: a function returns True if param
                                      will be excluded from weight decay
    """
    if len(skip_list) == 0 and not filter_bias_and_bn:
        exclude_from_weight_decay_fn = None
    else:
        skip_list_all = []
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
                skip_list_all.append(name)

        def exclude_fn(param):
            for name in skip_list_all:
                if param == name:
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
