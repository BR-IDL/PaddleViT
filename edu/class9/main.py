import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = np.arange(32).astype('float32')[:, np.newaxis]

    def __getitem__(self, idx):
        return paddle.to_tensor(self.data[idx]), paddle.to_tensor(self.data[idx])

    def __len__(self):
        return len(self.data)

def get_dataset():
    dataset = MyDataset()
    return dataset

def get_dataloader(dataset, batch_size):
    sampler = DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader

def build_model():
    model = nn.Sequential(*[
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8,10)])

    return model

def main_worker(*args):
    dataset = args[0]
    dataloader = get_dataloader(dataset, batch_size=1)
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    model = build_model()
    model = paddle.DataParallel(model)
    print(f'Hello PPViT, I am [{local_rank}]: I built a model for myself.')

    tensor_list = []
    for data in dataloader:
        sample = data[0]
        label = data[1]

        out = model(sample)
        out = out.argmax(1)
        print(f'[{local_rank}]: I got data: {sample.cpu().numpy()}, label: {sample.cpu().numpy()}, out: {out.cpu().numpy()}')

        dist.all_gather(tensor_list, out)
        if local_rank == 0:
            print(f'I am master ([{local_rank}]): I got all_gathered out: {tensor_list}')
        break

def main():
    dataset = get_dataset()
    dist.spawn(main_worker, args=(dataset, ), nprocs=8)

if __name__ == "__main__":
    main()
