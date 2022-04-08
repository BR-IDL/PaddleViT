# Scalable Vision Transformers with Hierarchical Pooling [arxiv](https://arxiv.org/abs/2103.10619) 

PaddlePaddle training/validation code and pretrained models for **HVT**.

The official pytorch implementation is [here](https://github.com/zhuang-group/HVT).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./hvt.png" alt="drawing" width="100%" height="80%"/>
<h4 align="center">HVT Model Overview</h4>
</p>

### Update 
- Update (2022-04-08): Code is updated.
- Update (2021-12-28): Code is released and ported weights are uploaded.

## Models Zoo
| Model          | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|----------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| HVT-Ti-1       | 69.45 | 89.28 | 5.7M    | 0.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/11BW-qLBMu_1TDAavlrAbfVlXB53dgm42/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16rZvJqL-UVuWFsCDuxFDqg?pwd=egds?pwd=egds) |
| HVT-S-0        | 80.30 | 95.15 | 22.0M   | 4.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1GlJ2j2QVFye1tAQoUJlgKTR_KELq3mSa/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L-tjDxkQx00jg7BsDClabA?pwd=hj7a?pwd=hj7a) |
| HVT-S-1        | 78.06 | 93.84 | 22.1M   | 2.4G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/16H33zNIpNrHBP1YhCq4zmLjRYQJ0XEmX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1quOsgVuxTcauISQ3SehysQ?pwd=tva8?pwd=tva8) |
| HVT-S-2        | 77.41 | 93.48 | 22.1M   | 1.9G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1U14LA7SXJtFep_SdUCjAV-cDOQ9A_OFk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nooWTBzaXyBtEgadn9VDmw?pwd=bajp?pwd=bajp) |
| HVT-S-3        | 76.30 | 92.88 | 22.1M   | 1.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/1m1CjOcZfPMLDRyX4QBgMhHV1m6rtu44v/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15sAOmQN6Hx0GLelYDuMQXw?pwd=rjch?pwd=rjch) |
| HVT-S-4        | 75.21 | 92.34 | 22.1M   | 1.6G   | 224        |  0.875   |  bicubic      |  [google](https://drive.google.com/file/d/14comGo9lO12dUeGGL52MuIJWZPSit7I0/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1o31hMRWR7FTCjUk7_fAOgA?pwd=ki4j?pwd=ki4j) |

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

For example, assume weight file is downloaded in `./hvt_s0_patch16_224.pdparams`, to use the `hvt_s0_patch16_224` model in python:
```python
from config import get_config
from hvt import build_hvt as build_model
# config files in ./configs/
config = get_config('./configs/hvt_s0_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./hvt_s0_patch16_224.pdparams')
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
-cfg='./configs/hvt_s0_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./hvt_s0_patch16_224.pdparams' \
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
-cfg='./configs/hvt_s0_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.







## Reference
```
@inproceedings{pan2021scalable,
  title={Scalable vision transformers with hierarchical pooling},
  author={Pan, Zizheng and Zhuang, Bohan and Liu, Jing and He, Haoyu and Cai, Jianfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={377--386},
  year={2021}
}
```
