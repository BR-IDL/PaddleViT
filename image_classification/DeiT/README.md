# Training data-efficient image transformers & distillation through attention, [arxiv](https://arxiv.org/abs/2012.12877) 

PaddlePaddle training/validation code and pretrained models for **DeiT**.

The official pytorch implementation is [here](https://github.com/facebookresearch/deit).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./deit.png" alt="drawing" width="60%" height="60%"/>
<h4 align="center">DeiT Model Overview</h4>
</p>

### Update 
- Update (2022-03-16): Code is refactored.
- Update (2021-09-27): More weights are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo

| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| deit_tiny_distilled_224   	| 74.52 | 91.90 | 5.9M    | 1.1G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1fku9-11O_gQI7UpZTjagVeND-pcHbV0C/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hAQ_85wWkqQ7sIGO1CmO9g?pwd=rhda) |
| deit_small_distilled_224  	| 81.17 | 95.41 | 22.4M   | 4.3G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1RIeWTdf5o6pwkjqN4NbW91GZSOCalI5t/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1wCVrukvwxISAGGjorPw3iw?pwd=pv28) |
| deit_base_distilled_224  		| 83.32 | 96.49 | 87.2M   | 17.0G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/12_x6-NN3Jde2BFUih4OM9NlTwe9-Xlkw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ZnmAWgT6ewe7Vl3Xw_csuA?pwd=5f2g) |
| deit_base_distilled_384  		| 85.43 | 97.33 | 87.2M   | 49.9G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1i5H_zjSdHfM-Znv89DHTv9ChykWrIt8I/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1PQsQIci4VCHY7l2tCzMklg?pwd=qgj2) |

| Teacher Model | Link |
| -- | -- |
| RegNet_Y_160  | [google](https://drive.google.com/file/d/1_nEYFnQqlGGqboLq_VmdRvV9mLGSrbyG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NZNhiO4xDfqHiRiIbk9BCA?pwd=gjsm)   |

> *The results are evaluated on ImageNet2012 validation set.

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

For example, assume weight file is downloaded in `./deit_base_patch16_224.pdparams`, to use the `deit_base_patch16_224` model in python:
```python
from config import get_config
from deit import build_vit as build_model
# config files in ./configs/
config = get_config('./configs/deit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./deit_base_patch16_224.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate DeiT model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/deit_tiny_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./deit_tiny_patch16_224.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the DeiT model on ImageNet2012 with distillation, run the following script using command line:
```shell
sh run_train_multi_distill.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/deit_tiny_distilled_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.

## Finetuning
To finetune the DeiT model on ImageNet2012, run the following script using command line:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_distill.py \
-cfg='./configs/deit_base_distilled_patch16_384.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./deit_base_distilled_patch16_224.pdparams' \
-amp
```
> Note: use `-pretrained` argument to set the pretrained model path, you may also need to modify the hyperparams defined in config file.



## Reference
```
@inproceedings{touvron2021training,
  title={Training data-efficient image transformers \& distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={International Conference on Machine Learning},
  pages={10347--10357},
  year={2021},
  organization={PMLR}
}
```