# VOLO: Vision Outlooker for Visual Recognition, [arxiv](https://arxiv.org/abs/2106.13112) 

PaddlePaddle training/validation code and pretrained models for **VOLO**.

The official pytorch implementation is [here](https://github.com/sail-sg/volo).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./volo.png" alt="drawing" width="100%" height="100%"/>
    <h4 align="center">VOLO Model Overview</h4>
</p>

### Update 
- Update (2021-09-27): More weights are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| volo_d1_224  					| 84.12 | 96.78 | 26.6M   | 6.6G   | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1kNNtTh7MUWJpFSDe_7IoYTOpsZk5QSR9/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1EKlKl2oHi_24eaiES67Bgw)(xaim) |
| volo_d1_384  					| 85.24 | 97.21 | 26.6M   | 19.5G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1fku9-11O_gQI7UpZTjagVeND-pcHbV0C/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1qZWoFA7J89i2aujPItEdDQ)(rr7p) |
| volo_d2_224  					| 85.11 | 97.19 | 58.6M   | 13.7G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1KjKzGpyPKq6ekmeEwttHlvOnQXqHK1we/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1JCK0iaYtiOZA6kn7e0wzUQ)(d82f) |
| volo_d2_384  					| 86.04 | 97.57 | 58.6M   | 40.7G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1uLLbvwNK8N0y6Wrq_Bo8vyBGSVhehVmq/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1e7H5aa6miGpCTCgpK0rm0w)(9cf3) |
| volo_d3_224  					| 85.41 | 97.26 | 86.2M   | 19.8G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1OtOX7C29fJ20ESKQnYGevp4euxhmXKAT/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1vhARtV2wfI6EFf0Ap71xwg)(a5a4) |
| volo_d3_448  					| 86.50 | 97.71 | 86.2M   | 80.3G  | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1lHlYhra1NNp0dp4NWaQ9SMNNmw-AxBNZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Q6KiQw4Vu1GPm5RF9_eycg)(uudu) |
| volo_d4_224  					| 85.89 | 97.54 | 192.8M  | 42.9G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/16oXN7xuy-mkpfeD-loIVOK95PfptHhpX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1PE83ZLd5evkKmHJ1V2KDsg)(vcf2) |
| volo_d4_448  					| 86.70 | 97.85 | 192.8M  | 172.5G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1N9-1OhPewA5TBR9CX5oA10obDS8e4Cfa/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1QoJ2Sqe1SK9hxbmV4uZiyg)(nd4n) |
| volo_d5_224  					| 86.08 | 97.58 | 295.3M  | 70.6G  | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1fcrvOGbAmKUhqJT-pU3MVJZQJIe4Qina/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nqDcXMW00v9PKr3RQI-g1w)(ymdg) |
| volo_d5_448  					| 86.92 | 97.88 | 295.3M  | 283.8G | 448        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1aFXEkpfLhmQlDQHUYCuFL8SobhxUzrZX/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1K4FBv6fnyMGcAXhyyybhgw)(qfcc) |
| volo_d5_512  					| 87.05 | 97.97 | 295.3M  | 371.3G | 512        | 1.15     | bicubic       | [google](https://drive.google.com/file/d/1CS4-nv2c9FqOjMz7gdW5i9pguI79S6zk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16Wseyiqvv0MQJV8wwFDfSA)(353h) |

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

For example, assume the downloaded weight file is stored in `./volo_d5_224.pdparams`, to use the `volo_d5_224` model in python:
```python
from config import get_config
from volo import build_volo as build_model
# config files in ./configs/
config = get_config('./configs/volo_d5_224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./volo_d5_224')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate VOLO model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/volo_d5_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./volo_d5_224'
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
    -cfg='./configs/volo_d5_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./volo_d5_224'
```

</details>

## Training
To train the VOLO model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg='./configs/volo_d5_224.yaml' \
  -dataset='imagenet2012' \
  -batch_size=32 \
  -data_path='/dataset/imagenet' \
```


<details>

<summary>
Run evaluation using multi-GPUs:
</summary>


```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/volo_d5_224.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>



## Visualization Attention Map
**(coming soon)**

## Reference
```
@article{yuan2021volo,
  title={Volo: Vision outlooker for visual recognition},
  author={Yuan, Li and Hou, Qibin and Jiang, Zihang and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2106.13112},
  year={2021}
}
```
