# PVTv2: Improved Baselines with Pyramid Vision Transformer, [arxiv](https://arxiv.org/abs/2106.13797) 

PaddlePaddle training/validation code and pretrained models for **PVTv2**.

The official pytorch implementation is [here](https://github.com/whai362/PVT).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./pvtv2.png" alt="drawing" width="60%" height="60%"/>
<h4 align="center">PVTv2 Model Overview</h4>
</p>


### Update 
- Update (2021-09-27): Model FLOPs and # params are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| pvtv2_b0 						| 70.47	| 90.16	| 3.7M    | 0.6G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1wkx4un6y7V87Rp_ZlD4_pV63QRst-1AE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1mab4dOtBB-HsdzFJYrvgjA)(dxgb) |
| pvtv2_b1 						| 78.70	| 94.49	| 14.0M   | 2.1G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/11hqLxL2MTSnKPb-gp2eMZLAzT6q2UsmG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Ur0s4SEOxVqggmgq6AM-sQ)(2e5m) |
| pvtv2_b2 						| 82.02	| 95.99	| 25.4M   | 4.0G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1-KY6NbS3Y3gCaPaUam0v_Xlk1fT-N1Mz/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FWx0QB7_8_ikrPIOlL7ung)(are2) |
| pvtv2_b2_linear 				| 82.06	| 96.04	| 22.6M   | 3.9G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1hC8wE_XanMPi0_y9apEBKzNc4acZW5Uy/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1IAhiiaJPe-Lg1Qjxp2p30w)(a4c8) |
| pvtv2_b3 						| 83.14	| 96.47	| 45.2M   | 6.8G   | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/16yYV8x7aKssGYmdE-YP99GMg4NKGR5j1/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ge0rBsCqIcpIjrVxsrFhnw)(nc21) |
| pvtv2_b4 						| 83.61	| 96.69	| 62.6M   | 10.0G  | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1gvPdvDeq0VchOUuriTnnGUKh0N2lj-fA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1VMSD_Kr_hduCZ5dxmDbLoA)(tthf) |
| pvtv2_b5 						| 83.77	| 96.61	| 82.0M   | 11.5G  | 224 	    | 0.875    | bicubic 	   | [google](https://drive.google.com/file/d/1OHaHiHN_AjsGYBN2gxFcQCDhBbTvZ02g/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ey4agxI2Nb0F6iaaX3zAbA)(9v6n) |
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

For example, assume the downloaded weight file is stored in `./pvtv2_b0.pdparams`, to use the `pvtv2_b0` model in python:
```python
from config import get_config
from pvtv2 import build_pvtv2 as build_model
# config files in ./configs/
config = get_config('./configs/pvtv2_b0.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./pvtv2_b0.pdparams')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate PVTv2 model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg=./configs/pvtv2_b0.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=./path/to/pretrained/model/pvtv2_b0  # .pdparams is NOT needed
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
    -cfg=./configs/pvtv2_b0.yaml \
    -dataset=imagenet2012 \
    -batch_size=16 \
    -data_path=/path/to/dataset/imagenet/val \
    -eval \
    -pretrained=/path/to/pretrained/model/pvtv2_b0  # .pdparams is NOT needed
```

</details>


## Training
To train the PVTv2 Transformer model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg=./configs/pvtv2_b0.yaml \
  -dataset=imagenet2012 \
  -batch_size=16 \
  -data_path=/path/to/dataset/imagenet/train
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
    -cfg=./configs/pvtv2_b0.yaml \
    -dataset=imagenet2012 \
    -batch_size=32 \
    -data_path=/path/to/dataset/imagenet/train
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@article{wang2021pvtv2,
  title={Pvtv2: Improved baselines with pyramid vision transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={arXiv preprint arXiv:2106.13797},
  year={2021}
}
```
