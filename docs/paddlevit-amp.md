English | [简体中文](./paddlevit-amp-cn.md)

# PaddleViT: How to use Automatic Mixed Precision (AMP) Training?

## Introduction:
PaddleViT supports AMP training for both single-gpu and multi-gpu settings. Briefly, automatic mixed precision (AMP) training is the process of using both full precision (a.k.a. FP32) and half precision (a.k.a FP16) during the model training. The aim is to speed up training while maintaining the accuracy. More information can be found in the Paddle AMP tutorial [here](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/01_paddle2.0_introduction/basic_concept/amp_en.html) and NVIDIA official website [here](https://developer.nvidia.com/automatic-mixed-precision).

> Note: only Nvidia Ampere, Volta and Turing GPUs are supported FP16 computing. 

## PaddleViT AMP training:
PaddleViT provides very simple implementations to enable amp training for vision tasks. For example, similar to the standard training script in image classification, adding the input argument `-amp` will switch to the amp training mode.

For single-GPU training:
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=64 \
-data_path='/dataset/imagenet' \
-amp
```

For multi-GPU training:

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
We test the training speed on a single GPU (Nvidia V100) for the ViT base model with and without `amp`, and the results are shown in the following table:
|         | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | Average | Speedup |
|---------|-----|-----|-----|-----|-----|-----|-----|---------|---------|
| AMP off | 78s | 78s | 79s | 78s | 78s | 79s | 78s | 78.29s  |    -    |
| AMP on  | 42s | 41s | 41s | 41s | 41s | 41s | 41s | 41.14s  | 1.903   |

In the above table, each item represents the training time in seconds for 100 training iterations. It can be see that the average training time is 78.29s/100iter, and 41.14s/100iter, for `amp off` and `amp on`, respectively. The training speed is increased by about **1.9** times.