# VOLO: Vision Outlooker for Visual Recognition, [arxiv](https://arxiv.org/abs/2103.17239) 

PaddlePaddle training/validation code and pretrained models for **VOLO**.

The official pytorch implementation is [here](https://github.com/sail-sg/volo).

This implementation is developed by [PPViT](https://github.com/xperzy/PPViT/tree/master).



<img src="./volo.png" alt="drawing" width="100%" height="100%"/>
<figcaption align = "center">VOLO Model Overview</figcaption>

### Update 
Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                          | Acc@1 | Acc@5 | Image Size | Crop_pct | Interpolation | Link        |
|--------------------------------|-------|-------|------------|----------|---------------|--------------|
| volo_d5_224_86.10              | 86.08 | 97.58 | 224        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1GBOBPCBJYZfWybK-Xp0Otn0N4NXpct0G/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1t9gPLRAOkdXaG55fVADQZg)(td49) |
| volo_d5_512_87.07              | 87.05 | 97.97 | 512        | 1.15     | bicubic       | [google](https://drive.google.com/file/d/1Phf_wHsjRZ1QrZ8oFrqsYuhDr4TXrVkc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1X-WjpNqvWva2M977jgHosg)(irik) |

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
