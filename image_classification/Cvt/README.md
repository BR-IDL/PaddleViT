# CvT: ntroducing Convolutions to Vision Transformers, [arxiv](https://arxiv.org/abs/2103.15808) 

PaddlePaddle training/validation code and pretrained models for **CvT**.

The official pytorch implementation is [here](https://github.com/microsoft/CvT/).


This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./cvt.png" alt="drawing" width="80%" height="80%"/>
    <h4 align="center">CvT Model Overview</h4>
</p>





### Update 
- Update (2021-12-14): Code is released and ported weights are uploaded.

## Models Zoo



## Requirements
- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0

## Data 
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

## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume the downloaded weight file is stored in `./RepMLP-Res50-light-224_train.pdparams`, to use the `RepMLP-Res50-light-224_train` model in python:
```python
from config import get_config
from resmlp_resnet import build_resmlp_resnet as build_model
# config files in ./configs/
config = get_config('./configs/repmlpres50_light_224_train.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./RepMLP-Res50-light-224_train')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate ResMLP model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/cvt-13-224x224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=128 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='CvT-13-224x224-IN-1k'
```

<details>

<summary>
Run evaluation using multi-GPUs:
</summary>


```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main_multi_gpu.py \
    -cfg='./configs/cvt-13-224x224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./CvT-13-224x224-IN-1k'
```

</details>

## Training
To train the ResMLP Transformer model on ImageNet2012 with single GPUs, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/repmlpres50_light_224_train.yaml' \
    -dataset='imagenet2012' \
    -batch_size=32 \
    -data_path='/dataset/imagenet' \
```

<details>
<summary>
Run training using multi-GPUs:
</summary>


```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/repmlpres50_light_224_train.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \ 
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@article{wu2021cvt,
title={CvT: Introducing Convolutions to Vision Transformers},
author={Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang},
journal={arXiv preprint arXiv:2103.15808},
year={2021}
}
```