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

"""Utilities"""

import copy
import pickle
import numpy as np
import paddle
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


def collate_fn(batch):
    """Collate function for batching samples
    
    Samples varies in sizes, here convert samples to NestedTensor which pads the tensor,
    and generate the corresponding mask, so that the whole batch is of the same size.

    """
    # eliminate invalid data (where boxes is [] tensor)
    old_batch_len = len(batch)
    batch = [x for x in batch if x[1]['boxes'].shape[0] != 0]
    # try refill empty sample by other sample in current batch
    #print('batch len = ', old_batch_len)
    #print('new batch len = ', len(batch))
    new_batch_len = len(batch)
    for i in range(new_batch_len, old_batch_len):
        batch.append(copy.deepcopy(batch[i%new_batch_len]))
    #print('batch = ', batch)
    #print('filled batch len = ', len(batch))
    batch = list(zip(*batch)) # batch[0]: data tensor, batch[1]: targets dict

    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for idx, item in enumerate(sublist):
            maxes[idx] = max(maxes[idx], item)
    return maxes


class NestedTensor():
    """Each NestedTensor has .tensor and .mask attributes, which are paddle.Tensors"""
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    """make the batch handle different image sizes
    
    This method take a list of tensors with different sizes,
    then max size is selected as the final batch size,
    smaller samples are padded with zeros(bottom-right),
    and corresponding masks are generated.

    """
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size # len is the num of images in this batch
    b, c, h, w  = batch_shape
    dtype = tensor_list[0].dtype
    data_tensor = paddle.zeros(batch_shape, dtype=dtype)
    mask = paddle.ones((b, h, w), dtype='int32')
    # zip has broadcast for tensor and mask
    #print('===== inside nested_tensor_from_tensor_list')
    # zip cannot used in paddle, which will create a new tensor. in pytorch it works well
    #for img, pad_img, m in zip(tensor_list, tensor, mask):
    #    pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
    #    m[: img.shape[0], :img.shape[1]] = 0
    for idx in range(b):
        s0 = tensor_list[idx].shape[0]
        s1 = tensor_list[idx].shape[1]
        s2 = tensor_list[idx].shape[2]
        # direct set value raise error in current env, we use numpy to bypass
        data_tensor[idx, : s0, : s1, : s2] = tensor_list[idx].cpu().numpy()
        #data_tensor[idx, : s0, : s1, : s2] = tensor_list[idx]
        mask[idx, : s1, : s2] = 0
    return NestedTensor(data_tensor, mask)


def reduce_dict(input_dict, average=True):
    """Impl all_reduce for dict of tensors in DDP"""
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(values, axis=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


@paddle.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [paddle.zeros([])]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype('float32').sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
