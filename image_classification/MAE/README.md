# Masked Autoencoders Are Scalable Vision Learners, [arxiv](https://arxiv.org/abs/2111.06377) 

PaddlePaddle training/validation code and pretrained models for **MAE**.

The official pytorch implementation is [here](https://github.com/facebookresearch/mae).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./mae.png" alt="drawing" width="90%"/>
<h4 align="center">MAE Model Overview</h4>
</p>


### Update 
- Update (2022-03-02): Code is refactored and bugs are fixed.
- Update (2022-02-15): Code is refactored and ported weights are uploaded.
- Update (2021-12-13): Code is released.

## Note:
Current Version requires extra packages installed: `paddlenlp`.
You can use the following command to install paddlenlp:
```shell
pip install paddlenlp
```
> Note: the reason to use paddlenlp is we found the AdamW in paddle cannot handle layer wise decay properly, instead the paddlenlp.ops.optimizer.AdamWLD works well in our case, so we import this op for temp fix.


## Models Zoo
| Finetuned Model               | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| mae_finetuned_vit_base        | 83.72 | 96.54 | 86.4M   | 17.0G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1txV3fWnu_Jr17tCCqk9e_pFeuh7GkmvU/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1rIV2lYHEIYhD0ScTxmMi5A?pwd=svaw)(svaw) |
| mae_finetuned_vit_large       | 85.95 | 97.57 | 304.1M  | 59.9G  | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1dzVWxQ0_XTKqKKpA3pSSVU57rT_g8nOe/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1zlqmA-_fqCNZiuKOPMTtQA?pwd=tp48)(tp48) |
| mae_finetuned_vit_huge        | 86.90 | 98.07 | 631.7M  | 162.5G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1xqjdPez4uG495w3akVbHbn4YqUB1Nmmk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/17z-NK-akSlvYJSRZkUU2CQ?pwd=1fds)(1fds) |
> *The results are evaluated on ImageNet2012 validation set.

| Pretrained Model              | Link         |
|-------------------------------|--------------|
| mae_pretrain_vit_base        | [google](https://drive.google.com/file/d/1K7ZEaDj1D56i7uTX46hSelf0Ydbpmtie/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1aFdDhA61-5lB9g6LoAlKoQ?pwd=3fu3)(3fu3) |
| mae_pretrain_vit_large       | [google](https://drive.google.com/file/d/1UagT3mz_cLHcjyIQfyyLOkXtJXda3UbS/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1UIZuA_3uk5v-AHX41rjd0A?pwd=9c3s)(9c3s) |
| mae_pretrain_vit_huge        | [google](https://drive.google.com/file/d/1Y1lIO_COL2vkz2YvrmYt2yI8iAiRNiPh/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1XN-WkiiICqQUXcmv44PUxw?pwd=vc42)(vc42) |

> Note: current model weighs are ported from official repo for paddle, our trainied model weights are coming soon.

## Notebooks
We provide a few notebooks in aistudio to help you get started:

**\*(coming soon)\***


## Requirements
- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.2.0
- [yacs](https://github.com/rbgirshick/yacs)>=0.1.8

## Data 
ImageNet2012 dataset is used in the following folder structure:
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

For example, assume the downloaded weight file is stored in `./vit_base_patch16_224.pdparams`, to use the `vit_base_patch16_224` model in python:
```python
from config import get_config
from transformer import build_transformer as build_model
# config files in ./configs/
config = get_config('./configs/vit_base_patch16_224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
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
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu_finetune.py \
    -cfg='./configs/vit_base_patch16_224_finetune.yaml' \
    -dataset='imagenet2012' \
    -batch_size=32 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./mae_finetuned_vit_base'
```


## Finetuning
To finetune the ViT model on ImageNet2012, run the following script using command line:

```shell
sh run_finetune_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_finetune.py \
    -cfg='./configs/vit_base_patch16_224_finetune.yaml' \
    -dataset='imagenet2012' \
    -batch_size=32 \
    -data_path='/dataset/imagenet' \
    -pretrained='./mae_pretrain_vit_base'
    -amp
```

## Linear probing
To finetune(linear probe) the ViT model on ImageNet2012, run the following script using command line:

```shell
sh run_linear_probe_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_linearprobe.py \
    -cfg='./configs/vit_base_patch16_224_linearprobe.yaml' \
    -dataset='imagenet2012' \
    -batch_size=32 \
    -data_path='/dataset/imagenet' \
    -pretrained='./mae_pretrain_vit_base'
    -amp
```

## Pretraining
To pretrain the ViT model on ImageNet2012, run the following script using command line:

```shell
sh run_pretrain_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu_pretrain.py \
-cfg='./configs/vit_base_patch16_224_pretrain.yaml' \
-dataset='imagenet2012' \
-batch_size=32 \
-data_path='/dataset/imagenet' \
-amp
```

> Note: it is recommended to train the MAE model on multi-node GPUs.

## Visualization Attention Map
**(coming soon)**

## Reference
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```
