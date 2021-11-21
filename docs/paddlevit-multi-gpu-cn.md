## PaddleViT: 如何使用多GPU ?
本文档介绍如何使用和如何实现多GPU（单结点）以在`PaddleViT`中训练和评估模型的方法。

`PaddleViT`实现基于`paddle.distributed` 包的多GPU方案，此外我们还提供了一些用于GPU间通信和数据传输的有用功能。

> 详细的官方 `paddle.distribued` 文档可见：[here](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/Overview_cn.html)


### 1. 如何使用多GPU进行训练/验证？
在`PaddleViT`中，多GPU使用方法简单明了。通常，使用一个脚本文件（例如，`run_train_multi.sh`）来运行实验。 这个`.sh`脚本通过命令行选项运行python文件（例如`main_multi_gpu.py`）。

例如，验证脚本 `run_eval_multi.sh` 调用带有多个参数的 `main_multi_gpu.py` :
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
在这个shell脚本中:
- `CUDA_VISIBLE_DEVICES` 设置将使用哪些 GPU.
- `batch_size` 设置在单个GPU上的batch_size .

通过运行以下shell脚本可以开始多GPU训练实验，例如：
```
$ sh run_train_multi.sh
```

### 2. PaddleViT中的多GPU方案是如何实现的?
#### STEP 0: 准备
我们在`PaddleViT`中使用`paddle.distributed` 包：
```python
import paddle.distributed as distt
```

我们介绍了在多GPU上训练/验证的基本概念和步骤：
- 启动多个子流程
- 每一个进程在1个GPU上运行
- 每个进程运行自己的训练/验证
- 数据集被拆分，每个进程处理整个数据集的一部分
- 在每个GPU上，前向过程应用于其自己的批处理数据。
- 收集并平均每个GPU上的梯度。
- 每次迭代的平均梯度在每个GPU上同步。
- 使用平均梯度在每个GPU上应用梯度下降。
- 跨所有GPU收集验证结果。
- GPU之间的通信基于`NCCL2`.


#### STEP 1: 创建 `main` 方法
定义一个`main`方法包含以下步骤：
1. 创建`dataset`和`dataloader`。（见第2步）
2. 获取并设置使用的GPU数量。
3. 为多GPU训练/验证启动多处理。

`main`方法可能类似于：
```python
def main():
    dataset_train = get_dataset(config, mode='train')
    dataset_val = get_dataset(config, mode='val')
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(dataset_train, dataset_val, ), nprocs=config.NGPUS)
```
其中
- `paddle.static.cuda_places()`获取当前环境中所有可用的GPU. 
- `dist.spawn` 启动 `multiprocessing`
- `main_worker` 包含完整的训练/验证过程。
- `args` 将数据集发送到所有子进程。
- `nprocs` 确定要启动的子进程的数量，将其设置为GPU的数量。

#### STEP 2: 创建 `dataset` 和 `dataloader`
1. Dataset

    `dataset` 的定义方式和使用单GPU时的方式相同.通常，你需要创建一个实现 `paddle.io.Dataset`的数据集类. 需要实现`__getitem__` 和 `__len__` 方法，用于读取数据并获取整个数据集的总长度。 

   在我们的多GPU方案中，我们在主进程中创建一个single `dataset` ,它将通过`dist.spawn`中的`args`（作为参数）传递给所有子进程。
2. Dataloader

    `dataloader` 定义了如何加载批处理数据，你可以创建一个 `paddle.io.DataLoader` ，将  `paddle.io.Dataset` 和  `DistributedBatchSampler` 作为其输入。其他常用的输入参数是  `batch_size(int)`, `shuffle(bool)` 和 `collate_fn`.

    对于多GPU方案， `DistributedBatchSampler` 用于将数据集拆分为 `num_replicas` 并为每个进程/GPU (`rank`)采样批处理数据.  例如：
    ```python
    sampler = DistributedBatchSampler(dataset,
                                    batch_size=batch_size,
                                    shuffle=(mode == 'train'))
    ```
    dataloader在每个进程中初始化（意味着您需要在`main_worker` 方法中初始化实例）， `num_replicas` 和 `rank` 将由分布式环境自动确定。

#### STEP 3: Multi-GPU 训练
在STEP1中，`dist.spawn` 中的第一个参数是 `main_worker`, 这是包含完整训练/验证过程的方法。
你可以理解 `main` 方法在主进程(master)上运行, 它启动了许多子进程(workers). 
这些子进程运行`main_worker`中定义的内容.

具体来说, 在 `main_worker` 中有以下内容:
1. 初始化分布式环境: `dist.init_paralel_env()`
2. (可选) 获取world-size: `dist.get_world_size()`
3. (可选) 获取当前rank: `dist.get_rank()`
4. 为多GPU准备模型: `model=paddle.DataParallel(model)`
5. 使用 `DistributedBatchSampler`获取dataloader
6. 训练 (与single-gpu相同)

#### STEP 4: Multi-GPU 验证
在用于验证的 `main_worker` 中，我们将有以下内容:
1. 初始化分布式环境: `dist.init_paralel_env()`
2. 为多GPU准备环境： `model=paddle.DataParallel(model)`
3. 使用 `DistributedBatchSampler`获取dataloader
4. 验证(同single-gpu)
5. 对于每次迭代, **在所有GPU上收集结果**

由于每个进程/GPU对其自己的批处理数据进行推理，我们必须收集这些结果以获取整体性能。在Paddle中， `paddle.distributed.all_reduce`跨多个GPU收集张量，可以在每次迭代中调用：
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
请注意，，默认 `all_reduce` 返回GPU之间张量值的`SUM`，因此我们除以`world_size`以获取平均值。

最后，可以使用 `AverageMeter` 将结果记录为使用单GPU:
```python
batch_size = paddle.to_tensor(image.shape[0])
dist.all_reduce(batch_size)
val_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])
val_acc1_meter.update(acc1.numpy()[0], batch_size.numpy()[0])
```

### 3. 高级应用
对于需要在`PaddleViT`中的多个GPU之间进行高级通信/数据传输的开发人员，我们为`reduce`dict 对象和`gather`任何(picklable) 对象编写了两种方法，而不仅仅是`paddle.Tensor`.

具体来说:

- `reduce_dict(input_dict, average=True)` 被定义为一个 `dict` 存储 key: 张量对, 如果 `average`设置为 `True`,  `all_reduce` 将对于字典的每个值上按world size进行 `average` 。如果 `average` 为 `False`, 则常规的 `sum` 操作将被应用于dict中的每个值.

- `all_gather(data)` 被定义为 `all_gather` 任何可选取的数据, 而不仅仅 `paddle.Tensor`. 输入是一个数据对象，输出是从每个rank手机的数据列表。

> 详细的实现可以在PaddleVIT `object_detection/DETR/utils.py`中找到。
