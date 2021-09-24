# CycleMLP: A MLP-like Architecture for Dense Prediction,[arXiv](https://arxiv.org/abs/2107.10224)

PaddlePaddle training/validation code and pretrained models for **CycleMLP**.

The official and 3rd party pytorch implementation are [here](https://github.com/ShoufaChen/CycleMLP).


This implementation is developed by [PPViT](https://github.com/xperzy/PPViT/tree/master).

<p align="center">
<img src="./cyclemlp.png" alt="drawing" width="70%" height="70%"/>
<h4 align="center">CycleMLP Model Overview</h4>
</p>



### Update 
Update (2021-09-24): Code is released and ported weights are uploaded.

## Models Zoo
| Model                          | Acc@1 | Acc@5 | Image Size | Crop_pct | Interpolation | Link        |
|--------------------------------|-------|-------|------------|----------|---------------|--------------|
| cyclemlp_b1 | 78.85 | 94.60 | 224        | 0.9   | bicubic       |  |
| cyclemlp_b2 | 81.58 | 95.81 | 224        | 0.9     | bicubic      |  |
| cyclemlp_b3 | 82.42 | 96.07 | 224        | 0.9     | bicubic      |  |
| cyclemlp_b4 | 82.96 | 96.33 | 224 | 0.875 | bicubic | |
| cyclemlp_b5 | 83.25 | 96.44 | 224 | 0.875 | bicubic | |

> *The results are evaluated on ImageNet2012 validation set.
> 
> Note: ViP weights are ported from [here](https://github.com/ShoufaChen/CycleMLP)



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

For example, assume the downloaded weight file is stored in `./cyclemlp_b1.pdparams`, to use the `cyclemlp_b1` model in python:
```python
from config import get_config
from vip import build_vip as build_model
# config files in ./configs/
config = get_config('./configs/cyclemlp_b1.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./cyclemlp_b1')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate ResMLP model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/cyclemlp_b1.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./cyclemlp_b1'
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
    -cfg='./configs/cyclemlp_b1.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./cyclemlp_b1'
```

</details>

## Training
To train the ResMLP Transformer model on ImageNet2012 with single GPUs, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg='./configs/cyclemlp_b1.yaml' \
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
    -cfg='./configs/cyclemlp_b1.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \ 
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@misc{hou2021vision,
    title={Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
    author={Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
    year={2021},
    eprint={2106.12368},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
