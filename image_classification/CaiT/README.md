# Going deeper with Image Transformers, [arxiv](https://arxiv.org/abs/2103.17239) 

PaddlePaddle training/validation code and pretrained models for **CaiT**.

The official pytorch implementation is [here](https://github.com/facebookresearch/deit).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./cait.png" alt="drawing" width="80%" height="80%"/>
    <h4 align="center">CaiT Model Overview</h4>
</p>


### Update 
- Update (2021-09-27): More weights are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| cait_xxs24_224                | 78.38 | 94.32 | 11.9M   | 2.2G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1LKsQUr824oY4E42QeUEaFt41I8xHNseR/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1YIaBLopKIK5_p7NlgWHpGA)(j9m8) |
| cait_xxs36_224                | 79.75 | 94.88 | 17.2M   | 33.1G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zZx4aQJPJElEjN5yejUNsocPsgnd_3tS/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1pdyFreRRXUn0yPel00-62Q)(nebg) |
| cait_xxs24_384                | 80.97 | 95.64 | 11.9M   | 6.8G   | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1J27ipknh_kwqYwR0qOqE9Pj3_bTcTx95/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1uYSDzROqCVT7UdShRiiDYg)(2j95) |
| cait_xxs36_384                | 82.20 | 96.15 | 17.2M   | 10.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/13IvgI3QrJDixZouvvLWVkPY0J6j0VYwL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1GafA8B6T3h_vtmNNq2HYKg)(wx5d) |
| cait_s24_224                  | 83.45 | 96.57 | 46.8M   | 8.7G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1sdCxEw328yfPJArf6Zwrvok-91gh7PhS/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1BPsAMEcrjtnbOnVDQwZJYw)(m4pn) |
| cait_xs24_384                 | 84.06 | 96.89 | 26.5M   | 15.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1zKL6cZwqmvuRMci-17FlKk-lA-W4RVte/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1w10DPJvK8EwhOCm-tZUpww)(scsv) |
| cait_s24_384                  | 85.05 | 97.34 | 46.8M   | 26.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1klqBDhJDgw28omaOpgzInMmfeuDa7NAi/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-aNO6c7Ipm9x1hJY6N6G2g)(dnp7) |
| cait_s36_384                  | 85.45 | 97.48 | 68.1M   | 39.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1m-55HryznHbiUxG38J2rAa01BYcjxsRZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-uWg-JHLEKeMukFFctoufg)(e3ui) |
| cait_m36_384                  | 86.06 | 97.73 | 270.7M  | 156.2G | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1WJjaGiONX80KBHB3YN8mNeusPs3uDhR2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1aZ9bEU5AycmmfmHAqZIaLA)(r4hu) |
| cait_m48_448                  | 86.49 | 97.75 | 355.8M  | 287.3G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lJSP__dVERBNFnp7im-1xM3s_lqEe82-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/179MA3MkG2qxFle0K944Gkg)(imk5) |

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

For example, assume the downloaded weight file is stored in `./cait_xxs24_224.pdparams`, to use the `cait_xxs24_224` model in python:
```python
from config import get_config
from cait import build_cait as build_model
# config files in ./configs/
config = get_config('./configs/cait_xxs24_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./cait_xxs24_224.pdparams')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate CaiT model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg=./configs/cait_xxs24_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=/path/to/pretrained/model/cait_xxs24_224  # .pdparams is NOT needed
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
    -cfg=./configs/cait_xxs24_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=/path/to/pretrained/model/cait_xxs24_224  # .pdparams is NOT needed
```

</details>

## Training
To train the CaiT Transformer model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg=./configs/cait_xxs24_224.yaml \
  -dataset=imagenet2012 \
  -batch_size=32 \
  -data_path=/path/to/dataset/imagenet/train \
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
    -cfg=./configs/cait_xxs24_224.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/train \
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@article{zhang2021gmlp,
  title={GMLP: Building Scalable and Flexible Graph Neural Networks with Feature-Message Passing},
  author={Zhang, Wentao and Shen, Yu and Lin, Zheyu and Li, Yang and Li, Xiaosen and Ouyang, Wen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2104.09880},
  year={2021}
}
```
