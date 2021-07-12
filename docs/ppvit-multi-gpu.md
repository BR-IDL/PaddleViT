## PPViT: How to use multi-gpu?
This document presents **how to use** and **how to implement** multi-gpu (single node) for training and validation in `PPViT` for training and validating your model. 

`PPViT` implements multi-gpu schemes based on `paddle.distributed` package and we also hack some useful functions for inter-gpu communication and data transfer.

> Detailed official `paddle.distribued` docs can be found: [here](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/Overview_cn.html)

### 1. How to use multi-gpu for training/validation?
In `PPViT`, multi-gpu is easy and straightforward to use. Typically, you will have a script file (e.g., `run_train_multi.sh`) to start your experiment. This `.sh` script runs the python file (e.g., `main_multi_gpu.py`) with commandline options. For example, a validation script `run_eval_multi.sh` calls the `main_multi_gpu.py` with a number of arguments:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./vit_base_patch16_224' \
```
In this shell script:
- `CUDA_VISIBLE_DEVICES` sets which gpus will be used.
- `batch_size` sets the batch_size on a **single** GPU.

You can run this shell script to start your experiment, e.g.:
```
$ sh run_train_multi.sh
```

### 2. How does the multi-gpu schemes implement in PPViT?
#### STEP 0: Preparation
We use `paddle.distributed` package in `PPViT`:
```python
import paddle.distributed as distt
```

We present the basic concepts and steps of train/validate on multi-gpus:
- Multiple subprocesses are launched.
- Each process runs on 1 single GPU.
- Each process runs its own training/validation.
- Dataset is splitted, each process handles one part of the whole dataset.
- On each GPU, forward is applied on its own batch data.
- Gradients on each GPU are collected and averaged (all_reduced).
- Averaged gradients are synced on each GPU for each iteration.
- Gradient descent is applied on each GPU using the averaged gradients.
- Validation results are collected across all GPUs.
- Communication between GPUs are based on `NCCL2`.


#### STEP 1: Create `main` method
Define a `main` method contains the following steps:
1. Create the `dataset` and `dataloader`. (see STEP2)
2. Get and set the number of GPUs to use.
3. Launch multi-processing for multi-gpu training/validation

The `main` method could be similar as:
```python
def main():
    dataset_train = get_dataset(config, mode='train')
    dataset_val = get_dataset(config, mode='val')
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(dataset_train, dataset_val, ), nprocs=config.NGPUS)
```
Where
- `paddle.static.cuda_places()` gets all the availabe GPUs in current env.
- `dist.spawn` launches `multiprocessing`
- `main_worker` contains the full training/validation procedures.
- `args` sends datasets to all the subprocesses.
- `nprocs` determines the number of subprocesses to launch, set this to the number of GPUs.

#### STEP 2: Create `dataset` and `dataloader`
1. Dataset

    `dataset` is defined in the same way as using single-GPU. Typically, you create a dataset class which implements `paddle.io.Dataset`. The `__getitem__` and `__len__` methods are required to implement, for reading the data and get the total length of the whole dataset.

    In our multi-gpu scheme, we create a single `dataset` in the main process which will pass (as arguments) to all the subprocesses by `args` in `dist.spawn` method.
2. Dataloader

    `dataloader` defineds how to load the batch data, you can create a `paddle.io.DataLoader` with a `paddle.io.Dataset` and a `DistributedBatchSampler` as its inputs. Other commonly used input parameters are `batch_size(int)`, `shuffle(bool)` and `collate_fn`.
    
    For multi-gpu scheme, the `DistributedBatchSampler` is used to split the dataset into `num_replicas` and sample batch data for each process/GPU (`rank`).  For example:
    ```python
    sampler = DistributedBatchSampler(dataset,
                                    batch_size=batch_size,
                                    shuffle=(mode == 'train'))
    ```
    The dataloader is initialized in each process (which means you will initialize the instance in `main_worker` method), the `num_replicas` and `rank` will be automated determined by the distributed env. 

#### STEP 3: Multi-GPU Training
In STEP1, the first argment in `dist.spawn` is `main_worker`, which is a method contains the full training/validation procedures. You can think the `main` method is run on the main process(master), which launches a number of subprocesses(workers). These subprocesses run the contents defined in `main_worker`.

Specifically, in the `main_worker` we have:
1. Init distributed env: `dist.init_paralel_env()`
2. (Optional) Get world-size: `dist.get_world_size()`
3. (Optional) Get current rank: `dist.get_rank()`
4. Make the model ready for multi-gpu: `model=paddle.DataParallel(model)`
5. Get dataloader with `DistributedBatchSampler`
6. Training (same as using single-gpu)

#### STEP 4: Multi-GPU Validation
In `main_worker` for validation, we will have:
1. Init distributed env: `dist.init_paralel_env()`
2. Make the model ready for multi-gpu: `model=paddle.DataParallel(model)`
5. Get dataloader with `DistributedBatchSampler`
4. Validation(same as single-gpu)
5. For each iteration, **gather the results across all GPUS**

Since each process/GPU runs inference for its own batch data, we must gather these results to get the overall performance. In paddle, `paddle.distributed.all_reduce` gathers the tensor across GPUs, which can be called in each iteration:
```python
output, _ = model(image) # inference
loss = criterion(output, label) # get loss

pred = F.softmax(output) # get perds
acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1)) # top1 acc
 
dist.all_reduce(loss) # gather loss from all GPUs
dist.all_reduce(acc1) # gather top1 acc from all GPUS
 
loss = loss / dist.get_world_size() # get average loss 
acc1 = acc1 / dist.get_world_size() # get average top1 acc
```
Note that default `all_reduce` returns the `SUM` of the tensor values across GPUs, therefore we divided the `world_size` to get the average.

Finally, the `AverageMeter` can be used to log the results as using single-gpu:
```python
batch_size = paddle.to_tensor(image.shape[0])
dist.all_reduce(batch_size)
val_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])
val_acc1_meter.update(acc1.numpy()[0], batch_size.numpy()[0])
```

### 3. Advanced functions
For developers who needs advanced communication/data transfer between GPUs in `PPViT`, we hacked two methods for `reduce` dict objects, and `gather` any (picklable) object rather than only `paddle.Tensor`.

Specifically:

- `reduce_dict(input_dict, average=True)` is defined to take an `dict` stores the key: tensor pairs, and if `average` is set to `True`, the `all_reduce` will conduct `average` by world size, on each value in the dict. If `average` is `False`, the regular `sum` operation will be applied on each value in the dict.

- `all_gather(data)` is defined to `all_gather` any pickable data, rather than only `paddle.Tensor`. The input is a data object, the output is a list of gathered data from each rank.

> Detailed implementations can be found in PPVIT `object_detection/DETR/utils.py`