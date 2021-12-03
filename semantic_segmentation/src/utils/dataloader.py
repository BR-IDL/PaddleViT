"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""

from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader


def get_dataloader(dataset,
                   shuffle=False,
                   batch_size=16,
                   drop_last=False,
                   num_workers=0,
                   num_iters=None,
                   start_iter=0):
    """
    get iterable data loader,
    the lenth is num_iters.
    """
    # make num_iters is valid
    if num_iters:
        assert num_iters > 0
    else:
        assert num_iters is None
    batch_sampler = DistributedBatchSampler(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=drop_last)
    if num_iters:
        batch_sampler = IterationBasedBatchSampler(batch_sampler=batch_sampler,
                                                   num_iterations=num_iters,
                                                   start_iter=start_iter)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers)
    return dataloader


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled.
    """
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        super(IterationBasedBatchSampler).__init__()
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            if hasattr(self.batch_sampler, "set_epoch"):
                self.batch_sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations
