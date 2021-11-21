"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""

from paddle.io import BatchSampler, DistributedBatchSampler, RandomSampler, SequenceSampler, DataLoader


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
