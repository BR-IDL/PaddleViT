English | [简体中文](./README_cn.md)

# PaddleViT-Classification: Visual Transformer and MLP Models for Image Classification
PaddlePaddle training/validation code and pretrained models for **Image Classification**.

This implementation is part of [PaddleViT](https://github.com/BR-IDL/PaddleViT.git) project.

## Update 
* Update (2021-12-23): Add BoTNet model.
* Update (2021-12-15): Add PoolFormer model.
* Update (2021-12-09): Add HaloNet model.
* Update (2021-12-08): Add PiT model.
* Update (2021-12-08): Add XCiT model.
* Update (2021-11-05): Update ConvMLP models.
* Update (2021-11-04): Update ConvMixer models.
* Update (2021-11-03): Update ViP models.
* Update (2021-10-28): Add MobileViT model.
* Update (2021-10-28): Add FocalTransformer model.
* Update (2021-10-28): Add CycleMLP model.
* Update (2021-10-19): Add BEiT model.
* Update (2021-10-12): Update code for training from scratch in Swin Transformer.
* Update (2021-09-28): Add AMP training.
* Update (2021-09-27): Add more ported model weights.
* Update (2021-09-09): Add FF-Only, RepMLP models.
* Update (2021-08-25): Init readme uploaded.

## Quick Start

 The following links are provided for the code and detail usage of each model architecture:
1. **[ViT](./ViT)**
2. **[DeiT](./DeiT)**
3. **[Swin](./SwinTransformer)**
4. **[VOLO](./VOLO)**
5. **[CSwin](./CSwin)**
6. **[CaiT](./CaiT)**
7. **[PVTv2](./PVTv2)**
8. **[Shuffle Transformer](./Shuffle_Transformer)**
9. **[T2T-ViT](./T2T_ViT)**
10. **[CrossViT](./CrossViT)**
10. **[Focal Transformer](./Focal_Transformer)**
11. **[BEiT](./BEiT)**
11. **[MobileViT](./MobileViT)**
11. **[ViP](./ViP)**
11. **[XCiT](./XCiT)**
11. **[PiT](./PiT)**
11. **[HaloNet](./HaloNet)**
11. **[PoolFormer](./PoolFormer)**
12. **[BoTNet](./BoTNet)**
13. **[MLP-Mixer](./MLP-Mixer)**
14. **[ResMLP](./ResMLP)**
15. **[gMLP](./gMLP)**
16. **[FF_Only](./FF_Only)**
17. **[RepMLP](./RepMLP)**
17. **[CycleMLP](./CycleMLP)**
17. **[ConvMixer](./ConvMixer)**
17. **[ConvMLP](./ConvMLP)**


## Installation
This module is tested on Python3.6+, and PaddlePaddle 2.1.0+. Most dependencies are installed by PaddlePaddle installation. You only need to install the following packages:
```shell
pip install yacs pyyaml
```
Then download the github repo:
```shell
git clone https://github.com/BR-IDL/PaddleViT.git
cd PaddleViT/image_classification
```
> Note: It is recommended to install the latest version of PaddlePaddle to avoid some CUDA errors for  PaddleViT training. For PaddlePaddle, please refer to this [link](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) for stable version installation and this [link](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html#gpu) for develop version installation. 

## Basic Usage
### Data Preparation
ImageNet2012 dataset is used in the following folder structure:
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
### Demo Example
To use the model with pretrained weights, go to the specific subfolder, then download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.  

Assume the downloaded weight file is stored in `./vit_base_patch16_224.pdparams`, to use the `vit_base_patch16_224` model in python:
```python
from config import get_config
from visual_transformer import build_vit as build_model
# config files in ./configs/
config = get_config('./configs/vit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./vit_base_patch16_224')
model.set_dict(model_state_dict)
```
> :robot: See the README file in each model folder for detailed usages.

## Basic Concepts
PaddleViT image classification module is developed in separate folders for each model with similar structure. Each implementation is around 3 type of classes and 2 types of scripts:
1. **Model classes** such as **[transformer.py](./ViT/transformer.py)**, in which the core *transformer model* and related methods are defined.
   
2. **Dataset classes** such as **[dataset.py](./ViT/datasets.py)**, in which the dataset, dataloader, data transforms are defined. We provided flexible implementations for you to customize the data loading scheme. Both single GPU and multi-GPU loading are supported.
   
3. **Config classes** such as **[config.py](./ViT/config.py)**, in which the model and training/validation configurations are defined. Usually, you don't need to change the items in the configuration, we provide updating configs by python `arguments` or `.yaml` config file. You can see [here](../docs/ppvit-config.md) for details of our configuration design and usage.
   
4. **main scripts** such as **[main_single_gpu.py](./ViT/main_single_gpu.py)**, in which the whole training/validation procedures are defined. The major steps of training or validation are provided, such as logging, loading/saving models, finetuning, etc. Multi-GPU is also supported and implemented in separate python script `main_multi_gpu.py`.
   
5. **run scripts** such as **[run_eval_base_224.sh](./ViT/run_eval_base_224.sh)**, in which the shell command for running python script with specific configs and arguments are defined.
   

## Model Architectures

PaddleViT now provides the following **transfomer based models**:
1. **[ViT](./ViT)** (from Google), released with paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
2. **[DeiT](./DeiT)** (from Facebook and Sorbonne), released with paper [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877), by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
3. **[Swin Transformer](./SwinTransformer)** (from Microsoft), released with paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030), by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
4. **[VOLO](./VOLO)** (from Sea AI Lab and NUS), released with paper [VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112), by Li Yuan, Qibin Hou, Zihang Jiang, Jiashi Feng, Shuicheng Yan.
5. **[CSwin Transformer](./CSwin)** (from USTC and Microsoft), released with paper [CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows
](https://arxiv.org/abs/2107.00652), by Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo.
6. **[CaiT](./CaiT)** (from Facebook and Sorbonne), released with paper [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239), by Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou.
7. **[PVTv2](./PVTv2)** (from NJU/HKU/NJUST/IIAI/SenseTime), released with paper [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797), by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.
8. **[Shuffle Transformer](./Shuffle_Transformer)** (from Tencent), released with paper [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/abs/2106.03650), by Zilong Huang, Youcheng Ben, Guozhong Luo, Pei Cheng, Gang Yu, Bin Fu.
9. **[T2T-ViT](./T2T_ViT)** (from NUS and YITU), released with paper [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
](https://arxiv.org/abs/2101.11986), by Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan.
10. **[CrossViT](./CrossViT)** (from IBM), released with paper [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899), by Chun-Fu Chen, Quanfu Fan, Rameswar Panda.
11. **[BEiT](./BEiT)** (from Microsoft Research), released with paper [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254), by Hangbo Bao, Li Dong, Furu Wei.
12. **[Focal Transformer](./Focal_Transformer)** (from Microsoft), released with paper [Focal Self-attention for Local-Global Interactions in Vision Transformers](https://arxiv.org/abs/2107.00641), by Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan and Jianfeng Gao.
13. **[Mobile-ViT](./MobileViT)** (from Apple), released with paper [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178), by Sachin Mehta, Mohammad Rastegari.
14. **[ViP](./ViP)** (from Oxford/ByteDance), released with [Visual Parser: Representing Part-whole Hierarchies with Transformers](https://arxiv.org/abs/2107.05790), by Shuyang Sun, Xiaoyu Yue, Song Bai, Philip Torr.
15. **[XCiT](./XCiT)** (from Facebook/Inria/Sorbonne), released with paper [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681), by Alaaeldin El-Nouby, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek, Hervé Jegou.
16. **[PiT](./PiT)** (from NAVER/Sogan University), released with paper [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302), by Byeongho Heo, Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Junsuk Choe, Seong Joon Oh.
17. **[HaloNet](./HaloNet)**, (from Google), released with paper [Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/abs/2103.12731), by Ashish Vaswani, Prajit Ramachandran, Aravind Srinivas, Niki Parmar, Blake Hechtman, Jonathon Shlens.11. 
18. **[PoolFormer](./PoolFormer)**, (from Sea AI Lab/NUS), released with paper [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418), by Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, Shuicheng Yan.
19. **[BoTNet](./BoTNet)**, (from UC Berkeley/Google), released with paper [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605), by Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, Ashish Vaswani.


PaddleViT now provides the following **MLP based models**:
1. **[MLP-Mixer](./MLP-Mixer)** (from Google), released with paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy
2. **[ResMLP](./ResMLP)** (from Facebook/Sorbonne/Inria/Valeo), released with paper [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404), by Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard Grave, Gautier Izacard, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, Hervé Jégou.
3. **[gMLP](./gMLP)** (from Google), released with paper [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050), by Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le.
4. **[FF Only](./FF_Only)** (from Oxford), released with paper [Do You Even Need Attention? A Stack of Feed-Forward Layers Does Surprisingly Well on ImageNet](https://arxiv.org/abs/2105.02723), by Luke Melas-Kyriazi.
5. **[RepMLP](./RepMLP)** (from BNRist/Tsinghua/MEGVII/Aberystwyth), released with paper [RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883), by Xiaohan Ding, Chunlong Xia, Xiangyu Zhang, Xiaojie Chu, Jungong Han, Guiguang Ding.
6. **[CycleMLP](./CycleMLP)** (from HKU/SenseTime), released with paper [CycleMLP: A MLP-like Architecture for Dense Prediction](https://arxiv.org/abs/2107.10224), by Shoufa Chen, Enze Xie, Chongjian Ge, Ding Liang, Ping Luo.
7. **[ConvMixer](./ConvMixer)** (from Anonymous), released with [Patches Are All You Need?](https://openreview.net/forum?id=TVHS5Y4dNvM), by Anonymous.
8. **[ConvMLP](./ConvMLP)** (from UO/UIUC/PAIR), released with [ConvMLP: Hierarchical Convolutional MLPs for Vision](https://arxiv.org/abs/2109.04454), by Jiachen Li, Ali Hassani, Steven Walton, Humphrey Shi.

#### Coming Soon: ####
1. **[CvT]()** (from McGill/Microsoft), released with paper [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808), by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang
2. **[HVT]()** (from Monash University), released with paper [Scalable Vision Transformers with Hierarchical Pooling](https://arxiv.org/abs/2103.10619), by Zizheng Pan, Bohan Zhuang, Jing Liu, Haoyu He, Jianfei Cai.
3. **[DynamicViT]()** (from Tsinghua/UCLA/UW), released with paper [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/abs/2106.02034), by Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh.


## Contact
If you have any questions, please create an [issue](https://github.com/BR-IDL/PaddleViT/issues) on our Github.
