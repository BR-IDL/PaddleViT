# PoolFormer: MetaFormer is Actually What You Need for Vision, [arxiv](https://arxiv.org/abs/2111.11418) 

PaddlePaddle training/validation code and pretrained models for **PoolFormer**.

The official TF implementation is [here](https://github.com/sail-sg/poolformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./poolformer.png" alt="drawing" width="70%"/>
<h4 align="center">PoolFormer Model Overview</h4>
</p>



### Update 
- Update (2021-12-10): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| poolformer_s12 | 77.24 | 93.51 | 11.9M | 1.8G | 224        | 0.9   | bicubic       |  |
| poolformer_s24 | 80.33 | 95.05 | 21.3M   | 3.4G  | 224        | 0.9      | bicubic       |      |
| poolformer_s36 | 81.43 | 95.45 | 30.8M   | 5.0G  | 224        | 0.9      | bicubic       |      |
| poolformer_m36 | 82.11 | 95.69 | 56.1M   | 8.9G  | 224        | 0.95     | bicubic       |      |
| poolformer_m48 | 82.46 | 95.96 | 73.4M   | 11.8G | 224        | 0.95     | bicubic       |      |

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

For example, assume the downloaded weight file is stored in `./poolformer_s12.pdparams`, to use the `poolformer_s12` model in python:
```python
from config import get_config
from poolformer import build_poolformer as build_model
# config files in ./configs/
config = get_config('./configs/poolformer_s12.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./poolformer_s12')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate PoolFormer model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/poolformer_s12.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./poolformer_s12'
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
    -cfg='./configs/poolformer_s12.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./poolformer_s12'
```

</details>


## Training
To train the PoolFormer model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg='./configs/poolformer_s12.yaml' \
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
    -cfg='./configs/poolformer_s12.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>



## Visualization Attention Map
**(coming soon)**

## Reference
```
@article{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2111.11418},
  year={2021}
}
```

