# PPViT #
Implementation of SOTA visual transformers and mlp models on PaddlePaddle 2.0+

## Introduction ##
PaddlePaddle Visual Transformers (`PPViT`) is a collection of PaddlePaddle image models beyond convolution, which are mostly based on visual transformers, visual attentions, and MLPs, etc. PPViT also integrates popular layers, utilities, optimizers, schedulers, data augmentations, training/validation scripts for PaddlePaddle 2.0+. The aim is to reproduce a wide variety of SOTA ViT models with full training/validation procedures.

## Models ##

### Image Classification ###
#### Now: ####
1. ViT ([An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929))
2. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
3. PVT ([Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122))
4. MLP-Mixer ([MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601))
5. ResMLP ([ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404))
6. gMLP ([Pay Attention to MLPs](https://arxiv.org/abs/2105.08050))
7. VOLO ([VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112))
8. CaiT ([Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239))

#### Coming Soon: ####
1. DeiT ([Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877))
2. T2T-ViT ([Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://arxiv.org/abs/2101.11986))
3. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
4. CSwin ([CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/pdf/2107.00652.pdf))
5. Focal Self-attention ([Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641))
6. HaloNet ([Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/abs/2103.12731))
7. Refined-ViT ([Refiner: Refining Self-attention for Vision Transformers](https://arxiv.org/pdf/2106.03714.pdf))
8. CrossViT ([CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899))



### Detection ###
#### Now: ####
1. DETR ([End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872))

#### Coming Soon: ####
1. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
2. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
3. PVT ([Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122))
4. Focal Self-attention ([Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641))
5. CSwin ([CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/pdf/2107.00652.pdf))
6. UP-DETR ([UP-DETR: Unsupervised Pre-training for Object Detection with Transformers](https://arxiv.org/abs/2011.09094))

### Semantic Segmentation ###
#### Now: ####
1. SETR ([Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840))

#### Coming Soon:  ####
1. FTN ([Fully Transformer Networks for Semantic Image Segmentation](https://arxiv.org/pdf/2106.04108.pdf))
2. Swin Transformer ([Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030))
3. Segmenter: ([Transformer for Semantic Segmentation](https://arxiv.org/pdf/2105.05633.pdf))
4. SegFormer ([SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203))
5. Shuffle Transformer ([Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650))
6. Focal Self-attention ([Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641))
5. CSwin ([CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows](https://arxiv.org/pdf/2107.00652.pdf))




### GAN ###
#### Coming Soon: ####
1. TransGAN ([TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074))
2. Styleformer ([Styleformer: Transformer based Generative Adversarial Networks with Style Vector](https://arxiv.org/abs/2106.07023))
3. ViTGAN ([ViTGAN: Training GANs with Vision Transformers](https://arxiv.org/pdf/2107.04589))




## Results (Ported Weights) ## 
### Image Classification ###
| Model                          | Acc@1 | Acc@5 | Image Size | Crop_pct | Interpolation | Model        |
|--------------------------------|-------|-------|------------|----------|---------------|--------------|
| vit_base_patch16_224           | 81.66 | 96.09 | 224        | 0.875    | bilinear      | [google](https://drive.google.com/file/d/13D9FqU4ISsGxWXURgKW9eLOBV-pYPr-L/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1kUQo1hhWZA0A1d7hNMvIYw)(nxhy) |
| vit_base_patch16_384           | 84.20 | 97.22 | 384        | 1.0      | bilinear      | [google](https://drive.google.com/file/d/1kWKaAgneDx0QsECxtf7EnUdUZej6vSFT/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1MW8Osbe4M70IPDNKygP9kQ)(8ack) |
| vit_large_patch16_224          | 83.00 | 96.47 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1jgwtmtp_cDWEhZE-FuWhs7lCdpqhAMft/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Jc8wrIydAsc-i2gL4DjztA)(g7ij) |
| swin_base_patch4_window7_224   | 85.27 | 97.56 | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1yjZFJoJeDFIfsxh9x10XGqCb8s2-Gtbp/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1j8Air9uFudq71S4FhogpWA)(ps9m) |
| swin_base_patch4_window12_384  | 86.43 | 98.07 | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1ThmGsTDZ8217-Zuo9o5EGLfzw8AI6N0w/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ZrTDMeKtr2Bm5uB63gxxQA)(ef9t) |
| swin_large_patch4_window12_384 | 87.14 | 98.23 | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1f30Mt80g5yLfEiViT4-kMLpyDjTUTV5B/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1RwKqdlR5N6BSZIptai7LrA)(5shn) |
| pvtv2_tiny_224                 | 70.47 | 90.16 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/13xboSs9W3rvFM-j5JK4MeXi-5IFyyo8O/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YHT8oDEtYgZDiDLss__YnA)(575w) |
| pvtv2_medium_224               | 82.02 | 95.99 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1B2ETlmmGqkxEIHkNg1Jq3h8NhJpUg9OZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XZC5YBBTCK7yUOXgDFhrSg)(ezfc) |
| pvtv2_large_224                | 83.77 | 96.61 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1QaRBbmKq2aPYLEKMxTUo4M1-8KdLzKR7/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vonKDOLvrIGjrf2BGh5ULA)(fbc4) |
| mixer_b16_224                  | 76.60 | 92.23 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1ZcQEH92sEPvYuDc6eYZgssK5UjYomzUD/view?usp=sharing)/[baidu](https://pan.baidu.com/s/12nZaWGMOXwrCMOIBfUuUMA)(xh8x) |
| resmlp_24_224                  | 79.38 | 94.55 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/15A5q1XSXBz-y1AcXhy_XaDymLLj2s2Tn/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nLAvyG53REdwYNCLmp4yBA)(jdcx) |
| gmlp_s16_224                   | 79.64 | 94.63 | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1TLypFly7aW0oXzEHfeDSz2Va4RHPRqe5/view?usp=sharing)/[baidu](https://pan.baidu.com/s/13UUz1eGIKyqyhtwedKLUMA)(bcth) |
| volo_d5_224_86.10              | 86.08 | 97.58 | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1GBOBPCBJYZfWybK-Xp0Otn0N4NXpct0G/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1t9gPLRAOkdXaG55fVADQZg)(td49) |
| volo_d5_512_87.07              | 87.05 | 97.97 | 512        | 1.15     | bicubic       | [google](https://drive.google.com/file/d/1Phf_wHsjRZ1QrZ8oFrqsYuhDr4TXrVkc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1X-WjpNqvWva2M977jgHosg)(irik) |
| cait_xxs24_224                 | 78.38 | 94.32 | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1LKsQUr824oY4E42QeUEaFt41I8xHNseR/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YIaBLopKIK5_p7NlgWHpGA)(j9m8) |
| cait_s24_384                   | 85.05 | 97.34 | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1GU0esukDvMg3u40FZB_5GiB6qpShjvGh/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1qvhNckJjcEf5HyVn8LuEeA)(qb86) |
| cait_m48_448                   | 86.49  | 97.75 | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lJSP__dVERBNFnp7im-1xM3s_lqEe82-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/179MA3MkG2qxFle0K944Gkg)(imk5) |


### Object Detection ###
| Model | backbone  | box_mAP | Model                                                                                                                                                       |
|-------|-----------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DETR  | ResNet50  | 42.0    | [google](https://drive.google.com/file/d/1ruIKCqfh_MMqzq_F4L2Bv-femDMjS_ix/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1J6lB1mezd6_eVW3jnmohZA)(n5gk) |
| DETR  | ResNet101 | 43.5    | [google](https://drive.google.com/file/d/11HCyDJKZLX33_fRGp4bCg1I14vrIKYW5/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1_msuuAwFMNbAlMpgUq89Og)(bxz2) |

### Semantic Segmentation ###
#### Pascal Context ####
|Model      | Backbone  | Batch_size | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint      |     ConfigFile  |
|-----------|-----------|------------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_large |     16     |   52.06   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1AUyBLeoAcMH0P_QGer8tdeU44muTUOCA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/11XgmgYG071n_9fSGUcPpDQ)(xdb8)   | [config](semantic_segmentation/configs/setr/SETR_Naive_Large_480x480_80k_pascal_context_bs_16.yaml) | 
|SETR_PUP   | ViT_large |     16     |   53.90   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1IY-yBIrDPg5CigQ18-X2AX6Oq3rvWeXL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1v6ll68fDNCuXUIJT2Cxo-A)(6sji) | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_480x480_80k_pascal_context_bs_16.yaml) |
|SETR_MLA   | ViT_Large |     8      |   54.39   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1utU2h0TrtuGzRX5RMGroudiDcz0z6UmV/view)/[baidu](https://pan.baidu.com/s/1Eg0eyUQXc-Mg5fg0T3RADA)(wora)| [config](semantic_segmentation/configs/setr/SETR_MLA_Large_480x480_80k_pascal_context_bs_8.yaml) |
|SETR_MLA   | ViT_large |     16     |   55.01   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/1SOXB7sAyysNhI8szaBqtF8ZoxSaPNvtl/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jskpqYbazKY1CKK3iVxAYA)(76h2) | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_480x480_80k_pascal_context_bs_16.yaml) |

#### Cityscapes ####
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_Large |     8      |     40k   |   76.71   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      | [google](https://drive.google.com/file/d/1QialLNMmvWW8oi7uAHhJZI3HSOavV4qj/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1F3IB31QVlsohqW8cRNphqw)(g7ro)  |  [config](semantic_segmentation/configs/setr/SETR_Naive_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_Naive | ViT_Large |     8      |     80k   |   77.31   |       -        | [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)      | [google](https://drive.google.com/file/d/1RJeSGoDaOP-fM4p1_5CJxS5ku_yDXXLV/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XbHPBfaHS56HlaMJmdJf1A)(wn6q)   |  [config](semantic_segmentation/configs/setr/SETR_Naive_Large_768x768_80k_cityscapes_bs_8.yaml)| 
|SETR_PUP   | ViT_Large |     8      |     40k   |   77.92   |       -        |  [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)     | [google](https://drive.google.com/file/d/12rMFMOaOYSsWd3f1hkrqRc1ThNT8K8NG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1H8b3valvQ2oLU9ZohZl_6Q)(zmoi)    | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_PUP   | ViT_Large |     8      |     80k   |   78.81   |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1tkMhRzO0XHqKYM0lojE3_g)(f793)    | [config](semantic_segmentation/configs/setr/SETR_PUP_Large_768x768_80k_cityscapes_bs_8.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     40k   |   76.70    |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1sUug5cMKSo6mO7BEI4EV_w)(qaiw)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_768x768_40k_cityscapes_bs_8.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     80k   |  77.26     |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1IqPZ6urdQb_0pbdJW2i3ow)(6bgj)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_768x768_80k_cityscapes_bs_8.yaml)| 


#### ADE20K ####
|Model      | Backbone  | Batch_size | Iteration | mIoU (ss) | mIoU (ms+flip) | Backbone_checkpoint | Model_checkpoint     |     ConfigFile  |
|-----------|-----------|------------|-----------|-----------|----------------|-----------------------------------------------|-----------------------------------------------------------------------|------------|
|SETR_Naive | ViT_Large |     16      |     160k   | 47.57   |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1_AY6BMluNn71UiMNZbnKqQ)(lugq)   | [config](semantic_segmentation/configs/setr/SETR_Naive_Large_512x512_160k_ade20k_bs_16.yaml)| 
|SETR_MLA   | ViT_Large |     8      |     160k   |  47.80   |       -        |   [google](https://drive.google.com/file/d/1TPgh7Po6ayYb1DksJeZp60LGnNyznr-r/view?usp=sharing)/[baidu](https://pan.baidu.com/s/18WSi8Jp3tCZgv_Vr3V1i7A)(owoj)    | [baidu](https://pan.baidu.com/s/1L83sdXWL4XT02dvH2WFzCA)(mrrv)    | [config](semantic_segmentation/configs/setr/SETR_MLA_Large_512x512_160k_ade20k_bs_8.yaml)| 

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

## Contributing ##
We encourage and appreciate your contribution to **PPViT** project, please refer to our workflow and work styles by [CONTRIBUTING.md](https://github.com/xperzy/PPViT/blob/develop/CONTRIBUTING.md)


## Licenses ##
#### Code #####
This repo is under the Apache-2.0 license. 

#### Pretrained Weights #####


## Citing #
>
