"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""
import paddle
import paddle.distributed as dist

from paddle.io import BatchSampler, DistributedBatchSampler, RandomSampler, SequenceSampler, DataLoader

__all__ = ['get_world_size', 'get_rank', 'synchronize', 'is_main_process',
           'reduce_dict', 'reduce_loss_dict', 'make_dataloader']


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with paddle.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = paddle.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with paddle.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = paddle.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def make_dataloader(dataset, shuffle, batchsize, distributed, num_workers, num_iters=None, start_iter=0):
    if distributed:
        data_sampler=DistributedBatchSampler(dataset, batch_size=batchsize, shuffle=True)
        dataloader = DataLoader(dataset, batch_sampler=data_sampler, num_workers=num_workers)

    if not distributed and shuffle:
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batchsize)
        if num_iters is not None:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    else:
        sampler = SequenceSampler(dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batchsize)
        if num_iters is not None:
            batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
        dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers)

    return dataloader


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
