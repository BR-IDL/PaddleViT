English | [简体中文](./README_cn.md)

# PaddleViT-GAN: Visual Transformer Models for GAN
PaddlePaddle training/validation code and pretrained models for **GAN**.

This implementation is part of [PaddleViT](https://github.com/BR-IDL/PaddleViT) project.

## Update 
Update (2021-08-25): Init readme uploaded.

## Quick Start

 The following links are provided for the code and detail usage of each model architecture:
1. **[Styleformer](./Styleformer)**
2. **[TransGAN](./transGAN)**


## Installation
This module is tested on Python3.6+, and PaddlePaddle 2.1.0+. Most dependencies are installed by PaddlePaddle installation. You only need to install the following packages:
```shell
pip install yacs pyyaml lmdb
```
Then download the github repo:
```shell
git clone https://github.com/xperzy/PPViT.git
cd PPViT/image_classification
```

## Basic Usage
### Data Preparation
**Cifar10**, **STL10**, **Celeba** and **LSUNchurch** datasets are used in the following folder structure:
#### [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html):
   
   We use `paddle.io.Dataset.Cifar10` to crate the Cifar10 dataset, download and prepare the data manually is NOT needed.
#### [STL10](https://cs.stanford.edu/~acoates/stl10/):
```
│STL10/
├── train_X.bin
│── train_y.bin
├── test_X.bin
│── test_y.bin
│── unlabeled.bin
```
#### [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
```
│Celeba/
├──img_align_celeba/
│  ├── 000017.jpg
│  │── 000019.jpg
│  ├── 000026.jpg
│  │── ......
```
#### [LSUN-church](https://www.yf.io/p/lsun):
```
│LSUNchurch/
├──church_outdoor_train_lmdb/
│  ├── data.mdb
│  │── lock.mdb
```
### Demo Example
For specific model example, go to the model folder, download the pretrained weight file, e.g., `./cifar10.pdparams`, to use the `styleformer_cifar10` model in python:
```python
from config import get_config
from generator import Generator
# config files in ./configs/
config = get_config('./configs/styleformer_cifar10.yaml')
# build model
model = Generator(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./cifar10')
model.set_dict(model_state_dict)
```

### Generate Sample Images
To generate sample images from pretrained models, download the pretrained weights, and run the following script using command line:
```shell
sh run_generate.sh
```
or 
```shell
python generate.py \
  -cfg='./configs/styleformer_cifar10.yaml' \
  -num_out_images=16 \
  -out_folder='./images_cifar10' \
  -pretrained='./cifar10.pdparams'
```
The output images are stored in `-out_folder` path.

> :robot: See the README file in each model folder for detailed usages.

## Basic Concepts
PaddleViT image classification module is developed in separate folders for each model with similar structure. Each implementation is around 3 type of classes and 2 types of scripts:
1. **Model classes** such as **[ViT_custom.py](./transGAN/models/ViT_custom.py)**, in which the core *transformer model* and related methods are defined.
   
2. **Dataset classes** such as **[dataset.py](./gan/transGAN/datasets.py)**, in which the dataset, dataloader, data transforms are defined. We provided flexible implementations for you to customize the data loading scheme. Both single GPU and multi-GPU loading are supported.
   
3. **Config classes** such as **[config.py](./gan/transGAN/config.py)**, in which the model and training/validation configurations are defined. Usually, you don't need to change the items in the configuration, we provide updating configs by python `arguments` or `.yaml` config file. You can see [here](../docs/ppvit-config.md) for details of our configuration design and usage.
   
4. **main scripts** such as **[main_single_gpu.py](./transGAN/main_single_gpu.py)**, in which the whole training/validation procedures are defined. The major steps of training or validation are provided, such as logging, loading/saving models, finetuning, etc. Multi-GPU is also supported and implemented in separate python script `main_multi_gpu.py`.
   
5. **run scripts** such as **[run_eval_cifar.sh](./transGAN/run_eval_cifar.sh)**, in which the shell command for running python script with specific configs and arguments are defined.
   

## Model Architectures

PaddleViT now provides the following **transfomer based models**:
1. **[TransGAN](./transGAN)** (from Seoul National University and NUUA), released with paper [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074), by Yifan Jiang, Shiyu Chang, Zhangyang Wang.
2. **[Styleformer](./Styleformer)** (from Facebook and Sorbonne), released with paper [Styleformer: Transformer based Generative Adversarial Networks with Style Vector](https://arxiv.org/abs/2106.07023), by Jeeseung Park, Younggeun Kim.



## Contact
If you have any questions, please create an [issue](https://github.com/BR-IDL/PaddleViT/issues) on our Github.
