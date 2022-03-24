import math
import random
import numpy as np
import paddle.distributed as dist
from paddle.io import Sampler, BatchSampler


class MultiScaleSamplerDDP(BatchSampler):
    def __init__(self,
                 base_image_width,
                 base_image_height,
                 base_batch_size,
                 num_data_samples,
                 min_scale_multi=0.5,
                 max_scale_multi=1.5,
                 num_scales=5,
                 is_train=False,
                 drop_last=False):
        super().__init__(drop_last=drop_last)
        min_image_width = int(base_image_width * min_scale_multi)
        min_image_height = int(base_image_height * min_scale_multi)
        max_image_width = int(base_image_width * max_scale_multi)
        max_image_height = int(base_image_height * max_scale_multi)

        world_size = dist.get_world_size()
        local_rank = dist.get_rank()

        local_num_samples = int(math.ceil(num_data_samples / world_size))
        total_size = local_num_samples * world_size
        image_indices = [idx for idx in range(num_data_samples)]
        image_indices += image_indices[:(total_size - num_data_samples)]
        assert len(image_indices) == total_size

        self.shuffle = False
        if is_train:
            width_dims = list(np.linspace(min_image_width, max_image_width, num_scales))
            height_dims = list(np.linspace(min_image_height, max_image_height, num_scales))
            width_dims = [(w // 32) * 32 for w in width_dims]
            height_dims = [(h // 32) * 32 for h in height_dims]

            image_batch_pairs = []
            base_elements = base_image_width * base_image_height * base_batch_size
            for (h, w) in zip(height_dims, width_dims):
                batch_size = max(1, int((base_elements / (h * w))))
                image_batch_pairs.append((h, w, batch_size))
            self.image_batch_pairs = image_batch_pairs
            self.shuffle = True
        else:
            self.image_batch_pairs = [(base_image_height, base_image_width, base_batch_size)]

        self.image_indices = image_indices
        self.local_num_samples = local_num_samples
        self.epoch = 0
        self.rank = local_rank
        self.world_size = world_size
        self.base_batch_size = base_batch_size

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch) # same as paper, all the gpus have the same size in each iter
            random.shuffle(self.image_indices)
            random.shuffle(self.image_batch_pairs)

        indices_local_rank = self.image_indices[self.rank:len(self.image_indices):self.world_size]

        start_index = 0
        while start_index < self.local_num_samples:
            h, w, batch_size = random.choice(self.image_batch_pairs)
            end_index = min(start_index + batch_size, self.local_num_samples)
            batch_indices = indices_local_rank[start_index: end_index]
            num_batch_samples = len(batch_indices)
            if num_batch_samples != batch_size:
                batch_indices += indices_local_rank[:(batch_size - num_batch_samples)]
            start_index += batch_size

            if len(batch_indices) > 0:
                batch = [(h, w, b_id) for b_id in batch_indices]
                yield batch

    def __len__(self):
        return self.local_num_samples // self.base_batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
