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
import math
import random
from PIL import Image
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist


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
    local_rank = dist.get_rank()
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
    ## console
    #sh = logging.StreamHandler(sys.stdout)
    #sh.setFormatter(logging.Formatter(log_format))
    #local_logger.addHandler(sh)

    # master_logger records avg performance
    if local_rank == 0:
        master_logger = logging.getLogger('master')
        # log.txt
        filename = os.path.join(file_path, 'log.txt')
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(fh)
        # console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(sh)
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
    if master_logger and dist.get_rank() == 0:
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
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce = x_reduce / world_size
        return x_reduce.item()
    return x


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


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


class RandomResizedCropAndInterpolationWithTwoPic:
    """hack from https://github.com/microsoft/unilm/blob/master/beit/transforms.py#L67"""
    def __init__(self,
                 size,
                 second_size=None,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(image, scale, ratio):
        area = img.size[0] * img.size[1]
        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, image):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)
