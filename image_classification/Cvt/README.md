# CvT: Introducing Convolutions to Vision Transformers, [arxiv](https://arxiv.org/abs/2103.15808) 

PaddlePaddle training/validation code and pretrained models for **CvT**.

The official pytorch implementation is [here](https://github.com/microsoft/CvT/).


This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./cvt.png" alt="drawing" width="80%" height="80%"/>
    <h4 align="center">CvT Model Overview</h4>
</p>


### Update 
- Update (2021-12-24): Code is released and ported weights are uploaded.


## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| CvT-13-224      | 81.59 | 95.72 | 20M    | 4.5G    | 224        | 0.875      | bicubic       | [google](https://drive.google.com/file/d/1r0fnHn1bRPmN0mi8RwAPXmD4utDyOxEf/view?usp=sharing)/[baidu](https://pan.baidu.com/s/13xNwCGpdJ5MVUi369OGl5Q)(vev9) |
| CvT-21-224      | 82.52 | 96.06 | 32M    | 7.1G    | 224        | 0.875      | bicubic       | [google](https://drive.google.com/file/d/18s7nRfvcmNdbRuEpTQe02AQE3Y9UWVQC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mOjbMNoQb7X3VJD3LV0Hhg)(t2rv) |
| CvT-13-384   	  | 83.00 | 96.36 | 20M    | 16.3G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1J0YYPUsiXSqyExBPtOPrOLL9c16syllg/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1upITRr5lNHLjbBJtIr-jdg)(wswt) |
| CvT-21-384   	  | 83.27 | 96.16 | 32M    | 24.9G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1tpXv_yYXtvyArlYi7AFcHUOqemhyMWHW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hXKi3Kb7mNxPFVmR6cdkMg)(hcem) |
| CvT-13-384-22k  | 83.26 | 97.09 | 20M    | 16.3G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/18djrvq422u1pGLPxNfWAp6d17F7C5lbP/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YYv5rKPmroxKCnzkesUr0g)(c7m9) |
| CvT-21-384-22k  | 84.91 | 97.62 | 32M    | 24.9G   | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1NVXd7vxVoRpL-21GN7nGn0-Ut0L0Owp8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1N3xNU6XFHb1CdEOrnjKuoA)(9jxe) |
| CvT-w24-384-22k | 87.58 | 98.47 | 277M   | 193.2G  | 384        | 1.0        | bicubic       | [google](https://drive.google.com/file/d/1M3bg46N4SGtupK8FcvAOE0jltOwP5yja/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1MNJurm8juHRGG9SAw3IOkg)(bbj2) |



> *The results are evaluated on ImageNet2012 validation set.


## Notebooks
We provide a few notebooks in aistudio to help you get started:

**\*(coming soon)\***


## Requirements
- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0
- [yacs](https://github.com/rbgirshick/yacs)>=0.1.8

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

For example, assume the downloaded weight file is stored in `./CvT-13-224x224-IN-1k.pdparams`, to use the `CvT-13-224x224-IN-1k` model in python:
```python
from config import get_config
from cvt import build_cvt as build_model
# config files in ./configs/
config = get_config('./configs/cvt-13-224x224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./CvT-13-224x224-IN-1k')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate CvT model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/cvt-13-224x224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./CvT-13-224x224-IN-1k'
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
CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
To train the CvT Transformer model on ImageNet2012 with single GPUs, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg='./configs/cvt-13-224x224.yaml' \
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
    -cfg='./configs/cvt-13-224x224.yaml' \
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