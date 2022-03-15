# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, [arxiv](https://arxiv.org/abs/2010.11929) 

PaddlePaddle training/validation code and pretrained models for **ViT**.

The official TF implementation is [here](https://github.com/google-research/vision_transformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./vit.png" alt="drawing" width="90%"/>
<h4 align="center">ViT Model Overview</h4>
</p>


### Update 
- Update (2022-03-15): Code is refactored, old weights link are updated, more weights are uploaded.
- Update (2021-09-27): More weights are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| vit_tiny_patch16_224          | 75.48 | 92.84 | 5.7M    |  1.1G   | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1WTNschmv8QhP6LzcVg0wfKRl6pvEhXC9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1cWgzFAq-bGEnr9bRVJvLcQ?pwd=3x5q) |
| vit_tiny_patch16_384          | 78.42 | 94.54 | 5.7M    |  3.3G   | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/1tNbT6sYSzwKPptiFJJbOPVQxNfHk6H-_/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nYmffkK7WQ_kyjCoy650Hg?pwd=xtkx) |
| vit_small_patch32_224         | 76.23 | 93.35 | 22.9M   |  1.1G   | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1fMSeBtrtSCLP0S8kKTUrQLnhtWomLwl_/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1C10pfIvMTgo-RutaF2BCnw?pwd=pahf) |
| vit_small_patch32_384         | 80.48 | 95.60 | 22.9M   |  3.3G   | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/1DOqoIoyXMuevREQTTE-TKs8FHuLhbPRE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1hjpekAkD5HB7D7ODT7ga7w?pwd=y6mh) |
| vit_small_patch16_224         | 81.40 | 96.15 | 22.0M   |  4.3G   | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1cDaBMIRPE0AjCG3jAUj50KBHn3AiEkJz/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1HUZkJAGH34OOZP_Ou5-RJA?pwd=hhm6) |
| vit_small_patch16_384         | 83.80 | 97.10 | 22.0M   |  12.7G  | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/100Ok2hAOggjWydWYF__D9fg1kaS3usPY/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1MMDeAhmOtfHSgREwVY0heQ?pwd=58f9) |
| vit_base_patch32_224          | 80.68 | 95.61 | 88.2M   |  4.4G   | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1w59BzxawQ6cI5WMmQqrjY4cv6UXm1KkL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1CNA95A2n4sKD85JpPCobjA?pwd=wijp) |
| vit_base_patch32_384          | 83.35 | 96.84 | 88.2M   |  12.7G  | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/1aobpfQiPWs4JNGKWOh7MYMesNicCapia/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1qAak8kJJf4EN-qlMi-ipXQ?pwd=7zx5) |
| vit_base_patch16_224          | 84.58 | 97.30 | 86.4M   |  17.0G  | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1yXUYa4_lh6hkvDjlYH-wuM38BZrx4XRD/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1owsqWvBzbBk0L_3FTTBgng?pwd=shyg) |
| vit_base_patch16_384          | 85.99 | 98.00 | 86.4M   |  49.8G  | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/1ZjlKYimCft1D2BYpqo0asGQ-SwhgBmyI/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1QooEkJ9K1K78zxGSzV9FGQ?pwd=ksut) |
| vit_large_patch32_384         | 81.51 | 96.09 | 306.5M  |  44.4G  | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/11SVJzq47P1gvREWRL-rcZE7pLatk0xE2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XQrhmZYKmpHUWQJx6XSswA?pwd=hpzj) |
| vit_large_patch16_224         | 85.81 | 97.82 | 304.1M  |  59.9G  | 224        | 0.875    | bicubic       |[google](https://drive.google.com/file/d/1e4Z1azRVKbIxHp-N8l0jjRiRFRiI9Cah/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nkdziPg646jRMACa88g4zw?pwd=76xn) |
| vit_large_patch16_384         | 87.08 | 98.30 | 304.1M  |  175.9G | 384        | 1.0      | bicubic       |[google](https://drive.google.com/file/d/1YkLQVjfCdjNH1sf2NYifRYjSVMHfdszV/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mv2ZTATWv-SKERUS1PQzBw?pwd=4w2u) |
| | | | | | | | | |

> *The results are evaluated on ImageNet2012 validation set.

> Note: old model weights may not be corrected loaded to current version, please download and use the current model weights.


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
- `train_list.txt`: list of relative paths and labels of training images. You can download it from: [google](https://drive.google.com/file/d/10YGzx_aO3IYjBOhInKT_gY6p0mC3beaC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1G5xYPczfs9koDb7rM4c0lA?pwd=a4vm)(a4vm)
- `val_list.txt`: list of relative paths and labels of validation images. You can download it from: [google](https://drive.google.com/file/d/1aXHu0svock6MJSur4-FKjW0nyjiJaWHE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1TFGda7uBZjR7g-A6YjQo-g?pwd=kdga)(kdga) 


## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume weight file is downloaded in `./vit_base_patch16_224.pdparams`, to use the `vit_base_patch16_224` model in python:
```python
from config import get_config
from vit import build_vit as build_model
# config files in ./configs/
config = get_config('./configs/vit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./vit_base_patch16_224.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate ViT model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/vit_tiny_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./vit_tiny_patch16_224.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the ViT model on ImageNet2012, run the following script using command line:
```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.

## Finetuning
To finetune the ViT model on ImageNet2012, run the following script using command line:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/vit_base_patch16_224.yaml' \
-dataset='imagenet2012' \
-batch_size=128 \
-data_path='/dataset/imagenet' \
-pretrained='./vit_base_patch16_224.pdparams' \
-amp
```
> Note: use `-pretrained` argument to set the pretrained model path, you may also need to modify the hyperparams defined in config file.



## Reference
```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
