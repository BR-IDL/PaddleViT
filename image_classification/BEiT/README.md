# BEiT: BERT Pre-Training of Image Transformers, [arxiv](https://arxiv.org/abs/2106.08254) 

PaddlePaddle training/validation code and pretrained models for **BEiT**.

The official and 3rd party pytorch implementation are [here](https://github.com/microsoft/unilm/tree/master/beit).


This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT).

<p align="center">
<img src="./beit.png" alt="drawing" width="90%" height="90%"/>
<h4 align="center">BEiT Model Overview</h4>
</p>



### Update 
- Update (2022-03-24): Code is refactored and bugs are fixed.
- Update (2021-10-19): Bug fix and weights links are updated.
- Update (2021-09-27): Code is released and ported weights are uploaded.

## Models Zoo

| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| beit_base_patch16_224   | 85.21 | 97.66 | 87M    | 12.7G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1lq5NeQRDHkIQi7U61OidaLhNsXTWfh_Z/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1pjblqaESqfXVrpgo58oR6Q?pwd=fshn) |
| beit_base_patch16_384   | 86.81 | 98.14 | 87M    | 37.3G   | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1wn2NS7kUdlERkzWEDeyZKmcRbmWL7TR2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WVbNjxuIUh514pKAgZZEzg?pwd=arvc) |
| beit_large_patch16_224  | 87.48 | 98.30 | 304M   | 45.0G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/11OR1FKxzfafqT7GzTW225nIQjxmGSbCm/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1bvhERVXN2TyRcRJFzg7sIA?pwd=2ya2) |
| beit_large_patch16_384  | 88.40 | 98.60 | 304M   | 131.7G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/10EraafYS8CRpEshxClOmE2S1eFCULF1Y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1H76G2CGLY3YmmYt4-suoRA?pwd=qtrn) |
| beit_large_patch16_512  | 88.60 | 98.66 | 304M   | 234.0G  | 512        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1xIIocftsB1PcDHZttPqLdrJ-G4Tyfrs-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WtTVK_Wvg-izaF0M6Gzw-Q?pwd=567v) |


> Note:
>
> The results are evaluated on ImageNet2012 validation set.
>
> These models have been fine-tuned (ImageNet 22k -> 1k), weights are ported from [here](https://github.com/microsoft/unilm/tree/master/beit)

> Note :
> We are developing the **pretraining using DALL-E**, which is not supported right now.

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

For example, assume weight file is downloaded in `./beit_base_patch16_224.pdparams`, to use the `beit_base_patch16_224` model in python:
```python
from config import get_config
from beit import build_beit as build_model
# config files in ./configs/
config = get_config('./configs/beit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./beit_base_patch16_224.pdparams')
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
-cfg='./configs/beit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./beit_base_patch16_224.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the model on ImageNet2012, run the following script using command line:
```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/beit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.


## Reference
```
@article{beit,
      title={{BEiT}: {BERT} Pre-Training of Image Transformers}, 
      author={Hangbo Bao and Li Dong and Furu Wei},
      year={2021},
      eprint={2106.08254},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
