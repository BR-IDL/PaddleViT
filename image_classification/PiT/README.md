# Rethinking Spatial Dimensions of Vision Transformers, [arxiv](https://arxiv.org/abs/2103.16302)

PaddlePaddle training/validation code and pretrained models for **PiT**.

The official pytorch implementation is [here](https://github.com/naver-ai/pit).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./pit.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center">PiT Model Overview</h4>
</p>


### Update 
* Update (2022-03-30): Code is refactored.
* Update (2021-12-08): Code is updated and ported weights are uploaded.
* Update (2021-11-13): Code is released.

## Models Zoo
| Model          | Acc@1 	| Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|----------------|----------|-------|---------|--------|------------|----------|---------------|--------------|
| pit_ti 	     | 72.91	| 91.40	| 4.8M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1bbeqzlR_CFB8CAyTUN52p2q6ii8rt0AW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Yrq5Q16MolPYHQsT_9P1mw?pwd=ydmi)  |
| pit_ti_distill | 74.54	| 92.10 | 5.1M    | 0.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1m4L0OVI0sYh8vCv37WhqCumRSHJaizqX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1RIM9NGq6pwfNN7GJ5WZg2w?pwd=7k4s)  |
| pit_xs 	     | 78.18    | 94.16 | 10.5M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1qoMQ-pmqLRQmvAwZurIbpvgMK8MOEgqJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15d7ep05vI2UoKvL09Zf_wg?pwd=gytu)  |
| pit_xs_distill | 79.31 	| 94.36 | 10.9M   | 1.1G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1EfHOIiTJOR-nRWE5AsnJMsPCncPHEgl8/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DqlgVF7U5qHfGD3QJAad4A?pwd=ie7s)  |
| pit_s  		 | 81.08 	| 95.33 | 23.4M   | 2.4G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1TDSybTrwQpcFf9PgCIhGX1t-f_oak66W/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Vk-W1INskQq7J5Qs4yphCg?pwd=kt1n)  |
| pit_s_distill  | 81.99 	| 95.79 | 24.0M   | 2.5G   | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1U3VPP6We1vIaX-M3sZuHmFhCQBI9g_dL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L7rdWmMW8tiGkduqmak9Fw?pwd=hhyc)  |
| pit_b   		 | 82.44 	| 95.71 | 73.5M	  | 10.6G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/1-NBZ9-83nZ52jQ4DNZAIj8Xv6oh54nx-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XRDPY4OxFlDfl8RMQ56rEg?pwd=uh2v)  |
| pit_b_distill  | 84.14 	| 96.86 | 74.5M   | 10.7G  | 224        | 0.9      | bicubic       |[google](https://drive.google.com/file/d/12Yi4eWDQxArhgQb96RXkNWjRoCsDyNo9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vJOUGXPtvC0abg-jnS4Krw?pwd=3e6g)  |
> *The results are evaluated on ImageNet2012 validation set.

| Teacher Model | Link |
| -- | -- |
| RegNet_Y_160  | [google](https://drive.google.com/file/d/1_nEYFnQqlGGqboLq_VmdRvV9mLGSrbyG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NZNhiO4xDfqHiRiIbk9BCA?pwd=gjsm)   |

## Data Preparation
ImageNet2012 dataset is used in the following file structure:
```
│imagenet/
├──train_list.txt
├──val_list.txt
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
- `train_list.txt`: list of relative paths and labels of training images. You can download it from: [google](https://drive.google.com/file/d/10YGzx_aO3IYjBOhInKT_gY6p0mC3beaC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1G5xYPczfs9koDb7rM4c0lA?pwd=a4vm?pwd=a4vm)
- `val_list.txt`: list of relative paths and labels of validation images. You can download it from: [google](https://drive.google.com/file/d/1aXHu0svock6MJSur4-FKjW0nyjiJaWHE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1TFGda7uBZjR7g-A6YjQo-g?pwd=kdga?pwd=kdga) 


## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume weight file is downloaded in `./pit_b_224.pdparams`, to use the `pit_b_224` model in python:
```python
from config import get_config
from pit import build_pit as build_model
# config files in ./configs/
config = get_config('./configs/pit_b_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./pit_b_224.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/pit_b_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./pit_b_224.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the model on ImageNet2012 with distillation, run the following script using command line:
```shell
sh run_train_multi_distill.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/pit_b_distilled_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.

## Finetuning
To finetune the model on ImageNet2012, run the following script using command line:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/pit_b_distilled_384.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./pit_b_distilled_224.pdparams' \
-amp
```
> Note: use `-pretrained` argument to set the pretrained model path, you may also need to modify the hyperparams defined in config file.




## Reference
```
@inproceedings{heo2021pit,
    title={Rethinking Spatial Dimensions of Vision Transformers},
    author={Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year={2021},
}
```
