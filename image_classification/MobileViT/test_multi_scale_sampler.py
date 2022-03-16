import paddle
import paddle.distributed as dist
from paddle.io import Dataset
from paddle.io import DataLoader
from multi_scale_sampler import MultiScaleSamplerDDP

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        w, h, idx = index
        #print(f'inside dummydataset(local_rank: {dist.get_rank()}): {index}')
        #data = paddle.randn([3, 224, 224]) 
        data = paddle.randn([3, int(w), int(h)]) 
        label = paddle.randn([1])
        return data, label

    def __len__(self):
        return 5000


def get_dataset():
    dataset = DummyDataset()
    return dataset

#def collate_fn(batch_data_list):
#    for batch_data in batch_data_list:
#        print('collate: ', batch_data[0].shape)
#
#    return paddle.to_tensor(batch_data_list[0]), paddle.to_tensor(batch_data_list[1])


def get_dataloader(dataset):
    sampler = MultiScaleSamplerDDP(224, 224, 4, 5000, is_train=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler,
                            #collate_fn=collate_fn,
                            num_workers=1)
    return dataloader


def main_worker(*args):
    dataset = args[0] 
    dist.init_parallel_env()
    dataloader = get_dataloader(dataset)

    local_rank = dist.get_rank()
    for batch_id, data in enumerate(dataloader):
        #print('local_rank = ', local_rank, ', batch_id =', batch_id)
        #print(data[0].shape, data[1].shape)
        #print('-----')
        #if batch_id ==10:
        #    break
        break


def main():
    dataset_val = get_dataset()
    ngpus = len(paddle.static.cuda_places()) 
    dist.spawn(main_worker, args=(dataset_val, ), nprocs=ngpus)

if __name__ == "__main__":
    main()

