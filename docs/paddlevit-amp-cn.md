简体中文 | [English](./paddlevit-amp.md)

# PaddleViT:如何使用自动混合精度(AMP)训练 ？

## Introduction:
PaddleViT对于单gpu和多gpu均支持AMP训练。简而言之，自动混合精度(AMP)训练是指在训练模型期间同时使用全精度(FP32)和半精度(FP16)的过程。目的在于保持准确性的同时也能加快训练速度。关于Paddle AMP的更多教程可以参考[here](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/01_paddle2.0_introduction/basic_concept/amp_en.html)以及NVIDIA官方网站[here](https://developer.nvidia.com/automatic-mixed-precision).

> 注意: 只有 Nvidia Ampere, Volta 以及 Turing GPUs 支持 FP16 计算. 


## PaddleViT AMP training:
PaddleViT提供了面向视觉任务的amp训练的简单易实现的方式。例如，类似于图像分类任务中的标准训练脚本，添加输入参数 `-amp` 即可切换到amp训练模式。

对于 single-GPU 训练:
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-amp
```

对于 multi-GPU 训练:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-amp
```

## Benchmark
我们在单GPU (Nvidia V100)上分别测试使用 `amp`和不使用 `amp`两种情况下的ViT基础模型的训练速度，结果如下表所示：

|         | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | Average | Speedup |
|---------|-----|-----|-----|-----|-----|-----|-----|---------|---------|
| AMP off | 78s | 78s | 79s | 78s | 78s | 79s | 78s | 78.29s  |    -    |
| AMP on  | 42s | 41s | 41s | 41s | 41s | 41s | 41s | 41.14s  | 1.903   |

在上表中，每一项代表100次训练迭代的训练时间（以秒为单位）。可以看出，`amp off` 和 `amp on` 的平均训练时间分别为 78.29 s/100iter 和 41.14 s/100iter，训练速度提升约**1.9**倍。
