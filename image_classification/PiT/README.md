# Rethinking Spatial Dimensions of Vision Transformers, [arxiv](https://arxiv.org/abs/2103.16302)

PaddlePaddle training/validation code and pretrained models for **PiT**.

The official pytorch implementation is [here](https://github.com/naver-ai/pit).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./pit.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center">PiT Model Overview</h4>
</p>


### Update 
* Update (2021-12-08): Code is updated and ported weights are uploaded.
* Update (2021-11-13): Code is released.

## Models Zoo
| Model          | Acc@1 	| Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|----------------|----------|-------|---------|--------|------------|----------|---------------|--------------|
| pit_ti 	     | 72.91	| 91.40	| 4.8M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1bbeqzlR_CFB8CAyTUN52p2q6ii8rt0AW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Yrq5Q16MolPYHQsT_9P1mw)(ydmi)  |
| pit_ti_distill | 74.54	| 92.10 | 5.1M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1m4L0OVI0sYh8vCv37WhqCumRSHJaizqX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1RIM9NGq6pwfNN7GJ5WZg2w)(7k4s)  |
| pit_xs 	     | 78.18    | 94.16 | 10.5M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1qoMQ-pmqLRQmvAwZurIbpvgMK8MOEgqJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15d7ep05vI2UoKvL09Zf_wg)(gytu)  |
| pit_xs_distill | 79.31 	| 94.36 | 10.9M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1EfHOIiTJOR-nRWE5AsnJMsPCncPHEgl8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DqlgVF7U5qHfGD3QJAad4A)(ie7s)  |
| pit_s  		 | 81.08 	| 95.33 | 23.4M   | 2.4G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1TDSybTrwQpcFf9PgCIhGX1t-f_oak66W/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Vk-W1INskQq7J5Qs4yphCg)(kt1n)  |
| pit_s_distill  | 81.99 	| 95.79 | 24.0M   | 2.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1U3VPP6We1vIaX-M3sZuHmFhCQBI9g_dL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L7rdWmMW8tiGkduqmak9Fw)(hhyc)  |
| pit_b   		 | 82.44 	| 95.71 | 73.5M	  | 10.6G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1-NBZ9-83nZ52jQ4DNZAIj8Xv6oh54nx-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XRDPY4OxFlDfl8RMQ56rEg)(uh2v)  |
| pit_b_distill  | 84.14 	| 96.86 | 74.5M   | 10.7G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/12Yi4eWDQxArhgQb96RXkNWjRoCsDyNo9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vJOUGXPtvC0abg-jnS4Krw)(3e6g)  |
> *The results are evaluated on ImageNet2012 validation set.

| Teacher Model | Link |
| -- | -- |
| RegNet_Y_160  | [google](https://drive.google.com/file/d/1_nEYFnQqlGGqboLq_VmdRvV9mLGSrbyG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NZNhiO4xDfqHiRiIbk9BCA)(gjsm)   |

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

For example, assume the downloaded weight file is stored in `./swin_base_patch4_window7_224.pdparams`, to use the `swin_base_patch4_window7_224` model in python:
```python
from config import get_config
from pit import build_pit as build_model
# config files in ./configs/
config = get_config('./configs/pit_ti.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./pit_ti')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate PiT model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/pit_ti.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./pit_ti'
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
    -cfg='./configs/pit_ti.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./pit_ti'
```

</details>


## Training
To train the PiT model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_singel_gpu.py \
  -cfg='./configs/pit_ti.yaml' \
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
    -cfg='./configs/pit_ti.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@inproceedings{heo2021pit,
    title={Rethinking Spatial Dimensions of Vision Transformers},
    author={Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year={2021},
}
```
