from paddle.io import BatchSampler
import math
import random
import numpy as np

class MultiScaleSamplerDDP(BatchSampler):
    def __init__(self, 
                 dataset,
                 base_im_w, 
                 base_im_h, 
                 batch_size,
                 min_scale_mult = 0.5, 
                 max_scale_mult = 1.5, 
                 n_scales = 5, 
                 num_replicas=None,
                 rank=None, 
                 shuffle = False,
                 is_training = True):
        self.dataset = dataset
        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle

        from paddle.fluid.dygraph.parallel import ParallelEnv
        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks
        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.epoch = 0
        # min. and max. spatial dimensions
        min_im_w, max_im_w = int(base_im_w * min_scale_mult), int(base_im_w * max_scale_mult)
        min_im_h, max_im_h = int(base_im_h * min_scale_mult), int(base_im_h * max_scale_mult)
        
        # Get the GPU and node related information
        num_replicas  = self.nranks
        rank = self.local_rank

        # adjust the total samples to avoid batch dropping
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / num_replicas))
        self.total_size = self.num_samples * num_replicas
        img_indices = [idx for idx in range(len(self.dataset))]
        assert len(img_indices) == self.total_size

        self.shuffle = False
        if is_training:
            # compute the spatial dimensions and corresponding batch size
            width_dims = list(np.linspace(min_im_w, max_im_w, n_scales))
            height_dims = list(np.linspace(min_im_h, max_im_h, n_scales))
            # ImageNet models down-sample images by a factor of 32.
            # Ensure that width and height dimensions are multiples are multiple of 32.
            width_dims = [(w // 32) * 32 for w in width_dims]
            height_dims = [(h // 32) * 32 for h in height_dims]

            img_batch_pairs = list()
            base_elements = base_im_w * base_im_h * batch_size
            for (h, w) in zip(height_dims, width_dims):
                batch_size = max(1, (base_elements / (h * w)))
                # img_batch_pairs.append((h, w, batch_size))
                img_batch_pairs.append((int(h), int(w), int(batch_size)))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
        else:
            self.img_batch_pairs = [(base_im_h , base_im_w , batch_size)]
        
        self.img_indices = img_indices
        self.n_samples_per_replica = self.num_samples
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_pairs)
            indices_rank_i = self.img_indices[self.rank : len(self.img_indices) : self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank : len(self.img_indices) : self.num_replicas]

        start_index = 0
        while start_index < self.n_samples_per_replica:
            curr_h, curr_w, curr_bsz = random.choice(self.img_batch_pairs)

            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz:
                    batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            if len(batch_ids) > 0:
                    batch = [(curr_h, curr_w, b_id) for b_id in batch_ids]
                    yield batch
    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
    def set_epoch(self, epoch: int):
        self.epoch = epoch
