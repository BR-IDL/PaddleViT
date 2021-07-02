# PPViT #
Implementation of SOTA visual transformers and mlp models on PaddlePaddle 2.0+

## Introduction ##
PaddlePaddle Visual Transformers (`PPViT`) is a collection of PaddlePaddle image models beyond convolution, which are mostly based on visual transformers, visual attentions, and MLPs, etc. PPViT also integrates popular layers, utilities, optimizers, schedulers, data augmentations, training/validation scripts for PaddlePaddle 2.0+. The aim is to reproduce a wide variety of SOTA ViT models with full training/validation procedures.

## Models ##

### Image Classification ###
#### Now: ####
1. ViT ([An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929))
2. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
3. MLP-Mixer ([MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601))
4. ResMLP ([ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404))
5. gMLP ([Pay Attention to MLPs](https://arxiv.org/abs/2105.08050))

#### Coming Soon: ####
1. DeiT ([Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877))
2. T2T-ViT ([Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986))
3. PVT ([Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122))
4. CaiT ([Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239))
5. VOLO ([VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112))
6. CrossViT ([CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899))
7. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
8. Refined-ViT ([Refiner: Refining Self-attention for Vision Transformers](https://arxiv.org/pdf/2106.03714.pdf))


### Detection ###
#### Now: ####
1. DETR ([End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872))

#### Coming Soon: ####
1. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
2. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
3. PVT ([Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122))

### Segmentation ###
#### Coming Soon:  ####
1. SETR ([Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840))
2. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
3. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
4. SegFormer ([SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203))


### GAN ###
#### Coming Soon: ####
1. TransGAN ([TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074))




## Results (Ported Weights) ## 
### Image Classification ###
| Model                          | Acc@1 | Acc@5 | Param # (M) | Image Size | Crop_pct | Interpolation |
|--------------------------------|-------|-------|-------------|------------|----------|---------------|
| vit_base_patch16_224           | 80.31 | 95.49 |             | 224        | 1.0      | bilinear      |
| vit_base_patch16_384           | 83.90 | 97.05 |             | 384        | 1.0      | bilinear      |
| vit_large_patch16_224          | 81.99 | 96.00 |             | 224        | 1.0      | bilinear      |
| swin_base_patch4_window7_224   | 82.77 | 95.99 |             | 224        | 1.0      | bilinear      |
| swin_base_patch4_window12_384  | 85.54 | 97.10 |             | 384        | 0.9      | bilinear      |
| swin_large_patch4_window12_384 | 86.74 | 97.52 |             | 384        | 0.9      | bicubic       |
### Object Detection ###
| Model | backbone  | box_mAP |
|-------|-----------|---------|
| DETR  | ResNet50  | 42.0    |
| DETR  | ResNet101 | 43.5    |
### Segmentation ###
### GAN ###

## Results (Self-Trained Weights) ## 
### Image Classification ###
### Object Detection ###
### Segmentation ###
### GAN ###




## Validation Scripts ##
### Run on single GPU: ###

`sh run_eval.sh`

 or you can run the python script:

 `python main_single_gpu.py`

 with proper settings.

The script `run_eval.sh` calls the main python script `main_single_gpu.py` with a number of options, usually you need to change the following settings, e.g., for ViT base model:
```shell
python main_single_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2021' \
-data_path='/dataset/imagenet' \
-batch_size=128 \
-eval \
-pretrained='./vit_base_patch16_224'
```
> Note:
> - The `-pretrained` option accepts the path of pretrained weights file **without** the file extension (.pdparams).

### Run on multi GPU: ###

`sh run_eval_multi.sh`

 or you can run the python script:

 `python main_multi_gpu.py`

 with proper settings.

The script `run_eval_multi.sh` calls the main python script `main_multi_gpu.py` with a number of options, usually you need to change the following settings, e.g., for ViT base model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2021' \
-data_path='/dataset/imagenet' \
-batch_size=128 \
-eval \
-pretrained='./vit_base_patch16_224'
-ngpus=8
```
> Note:
>
> - that the `-pretrained` option accepts the path of pretrained weights file **without** the file extension (.pdparams).
>
> - If `-ngpu` is not set, all the available GPU devices will be used.


## Training Scripts ##
### Train on single GPU: ###

`sh run_train.sh`

 or you can run the python script:

 `python main_single_gpu.py`

 with proper settings.

The script `run_train.sh` calls the main python script `main_single_gpu.py` with a number of options, usually you need to change the following settings, e.g., for ViT base model:
```shell
python main_single_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2021' \
-data_path='/dataset/imagenet' \
-batch_size=128 \
```
> Note:
> - The training options such as lr, image size, model layers, etc., can be changed in the `.yaml` file set in `-cfg`. All the available settings can be found in `./config.py`

### Run on multi GPU: ###

`sh run_train_multi.sh`

 or you can run the python script:

 `python main_multi_gpu.py`

 with proper settings.

The script `run_train_multi.sh` calls the main python script `main_multi_gpu.py` with a number of options, usually you need to change the following settings, e.g., for ViT base model:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2021' \
-data_path='/dataset/imagenet' \
-batch_size=128 \
-ngpus=8
```
> Note:
>
> - The training options such as lr, image size, model layers, etc., can be changed in the `.yaml` file set in `-cfg`. All the available settings can be found in `./config.py`
> - If `-ngpu` is not set, all the available GPU devices will be used.


## Features ##
* optimizers
* Schedulers
* DDP
* Data Augumentation
* DropPath


## Licenses ##
### Code ###
### Pretrained Weights #####


## Citing #
>
