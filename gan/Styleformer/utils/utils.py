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
import pickle
import random
import numpy as np
import paddle
from paddle.optimizer.lr import LRScheduler
import paddle.distributed as dist
from paddle.optimizer.lr import LRScheduler


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


def all_gather(data):
    """ run all_gather on any picklable data (do not requires tensors)
    Args:
        data: picklable object
    Returns:
        data_list: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data) #write data into Bytes and stores in buffer
    np_buffer = np.frombuffer(buffer, dtype=np.int8)
    tensor = paddle.to_tensor(np_buffer, dtype='int32') # uint8 doese not have many ops in paddle

    # obtain Tensor size of each rank
    local_size = paddle.to_tensor([tensor.shape[0]])
    size_list = []
    dist.all_gather(size_list, local_size)
    max_size = max(size_list)

    # receiving tensors from all ranks,
    # all_gather does not support different shape, so we use padding
    tensor_list = []
    if local_size != max_size:
        padding = paddle.empty(shape=(max_size - local_size, ), dtype='int32')
        tensor = paddle.concat((tensor, padding), axis=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.astype('uint8').cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

# GP
def gradient_penalty(discriminator, real, fake):
    """gradient penalty"""
    # BATCH_SIZE,C,H,W = real.shape
    # the OP returns a random Tensor whose value is uniformly distributed
    # within the range [min, max), with a shape of shape and a data type of dtype.
    # epsilon ∼ U[0, 1].
    epsilon = paddle.randn((real.shape[0], 1, 1, 1)).cuda()
    # extend epsilon to real shape size
    # x_hat = real * epsilon + fake * (1 - epsilon), Picture after interpolation
    interpolated_images = paddle.to_tensor((real * epsilon + fake * (1 - epsilon)),
                                           stop_gradient=False)
    # the interpolated picture calculates the discriminator score
    mixed_scores = discriminator(interpolated_images)
    # print(mixed_scores)
    # fake = paddle.to_tensor(paddle.ones((real.shape[0], 1)), stop_gradient=True).cuda()
    fake = paddle.ones((real.shape[0], 1))
    # calculate the blend gradient on the interpolated graph
    # paddle.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
    # only_inputs = True, allow_unused=False, no_grad_vars=None)
    # for each input, calculate the gradient sum of all outputs relative to it
    gradient = paddle.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # try to make gradient points and flatten the gradient
    gradient = paddle.reshape(gradient, (gradient.shape[0], -1))
    # L2 norm
    gradient_norm = gradient.norm(2, axis=1)
    # calculate gradient_penalty
    gp = paddle.mean((gradient_norm - 1) ** 2)
    return gp

def DiffAugment(x, policy='', channels_first=True, affine=None):
    if policy:
        if not channels_first:
            x = x.transpose(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, affine=affine)
        if not channels_first:
            x = x.transpose(0, 2, 3, 1)
    return x

# belong to DiffAugment
def rand_brightness(x, affine=None):
    x = x + (paddle.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x

# belong to DiffAugment
def rand_saturation(x, affine=None):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (paddle.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2)
    x = x + x_mean
    return x

# belong to DiffAugment
def rand_contrast(x, affine=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (paddle.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5)
    x = x + x_mean
    return x

# belong to DiffAugment
def rand_cutout(x, ratio=0.5, affine=None):
    if random.random() < 0.3:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = paddle.randint(0,
                                  x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1],
                                  device=x.device)
        offset_y = paddle.randint(0,
                                  x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1],
                                  device=x.device)
        grid_batch, grid_x, grid_y = paddle.meshgrid(
            paddle.arange(x.size(0), dtype=paddle.long, device=x.device),
            paddle.arange(cutout_size[0], dtype=paddle.long, device=x.device),
            paddle.arange(cutout_size[1], dtype=paddle.long, device=x.device),
        )
        grid_x = paddle.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = paddle.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        del offset_x
        del offset_y
        mask = paddle.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        del mask
        del grid_x
        del grid_y
        del grid_batch
    return x

# belong to DiffAugment
def rand_translation(x, ratio=0.2, affine=None):
    shift_x, shift_y = int(x.shape[2] * ratio + 0.5), int(x.shape[3] * ratio + 0.5)
    translation_x = paddle.randint(-shift_x, shift_x + 1, shape=[x.shape[0], 1, 1])
    translation_y = paddle.randint(-shift_y, shift_y + 1, shape=[x.shape[0], 1, 1])
    grid_batch, grid_x, grid_y = paddle.meshgrid(
        paddle.arange(x.shape[0]),
        paddle.arange(x.shape[2]),
        paddle.arange(x.shape[3]),
    )
    grid_x = paddle.clip(grid_x + translation_x + 1, 0, x.shape[2] + 1)
    grid_y = paddle.clip(grid_y + translation_y + 1, 0, x.shape[3] + 1)
    x_pad = paddle.nn.functional.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.transpose([0, 2, 3, 1])[grid_batch, grid_x, grid_y].transpose([0, 3, 1, 2])
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
