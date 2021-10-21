# Focal Self-attention for Local-Global Interactions in Vision Transformers, [arxiv](https://arxiv.org/pdf/2107.00641)


PaddlePaddle training/validation code and pretrained models for Focal Transformer.

The official pytorch implementation is [here](https://github.com/microsoft/Focal-Transformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<div align=center>
  <img src='./model.png'>
  </div>

<center> <h3>Focal Transformer Model Overview</h3> </center>

### Update

- Update(2021-10-21): 
	- Code is released and ported weights are uploaded.
   - Support single gpu and multi gpus training.

## Models Zoo

| Model | UseConv | official-Acc@1 | paddle-Acc@1 | official-Acc@5 | paddle-Acc@5 | #Params | FLOPs | Image Size | Crop_pct | Interpolation | Link |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Focal-T     | No     | 82.2     | 82.03     | 95.9     | 95.86     | 28.9M     | 4.9G     | 224     | 0.875     | bicubic     | [google]()/[baidu]()     |
| Focal-T     | Yes     | 82.7     | 82.70     | 96.1     | 96.14     | 30.8M     | 4.9G     | 224     | 0.875      | bicubic     | [google]()/[baidu]()     |
| Focal-S     | No     | 83.6     | 83.55    | 96.2     | 96.29     | 51.1M     | 9.4G     | 224     | 0.875      | bicubic     | [google]()/[baidu]()     |
| Focal-S     | Yes     | 83.8     | 83.85     | 96.5     | 96.47     | 53.1M     | 9.4G     | 224     | 0.875      | bicubic     | [google]()/[baidu]()     |
| Focal-B     | No     | 84.0     | 83.98     | 96.5     | 96.48     | 89.8M     | 16.4G     | 224     | 0.875      | bicubic     | [google]()/[baidu]()     |
| Focal-B     | Yes     | 84.2     | **     | 97.1     | **     | 93.3M     | 16.4G     | 224     | 0.875      | bicubic     | [google]()/[baidu]()     |

> *The results are evaluated on ImageNet2012 validation set.

### Models trained from scratch using PaddleViT
(coming soon)

## Notebooks
We provide a few notebooks in aistudio to help you get started:

**(coming soon)**

## Requirements
- Python>=3.6
- yaml>=0.2.5
- PaddlePaddle>=2.1.0
- yacs>=0.1.8

## Data
`ImageNet2012 dataset` is used in the following folder structure:
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
To use the model with pretrained weights, download the .pdparam weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume the downloaded weight file is stored in `./focal_tiny_patch4_window7_224.pdparams`, to use the `focal_tiny_patch4_window7_224` model in python:

```python
from config import get_config
from focal_transformer import build_focal as build_model
# config files in ./configs/
config = get_config('./configs/focal_tiny_patch4_window7_224.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./focal_tiny_patch4_window7_224')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate `Focal Transformer` model performance on ImageNet2012 with a `single GPU`, run the following script using command line:

```shell
sh run_eval.sh
```

or

```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=64 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./focal_tiny_patch4_window7_224'
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
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=32 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./focal_tiny_patch4_window7_224'
```

</details>

## Training
To train the `Focal Transformer` model on ImageNet2012 with `single GPU`, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=32 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -output='./output'
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
python main_single_gpu.py \
    -cfg='./configs/focal_tiny_patch4_window7_224.yaml' \
    -dataset='imagenet2012' \
    -num_classes=1000 \
    -batch_size=4 \
    -image_size=224 \
    -data_path='/dataset/imagenet' \
    -output='./output'
```

</details>

## Arg Info
- *`-cfg`*: 模型config的yaml文件，位于`./configs`.
- *`-dataset`*: 使用预置的dataset.py中的数据加载类中预置数据加载方式——支持`imagenet2012`, `cifar10`, `cifar100`.
- *`-data_path`*: 数据集路径
- `-batch_size`: 训练数据的批大小，默认为`32`.
- `-image_size`: 输入网络图像的大小，默认为`224`.
- `-num_classes`: 模型分类数，默认为`1000`——数据集不同时，要更正同使用的数据集一致.
- `-output`: 模型文件保存的输出目录，默认为`./output`.
- `-pretrained`: 模型的预训练模型路径(不必加`.pdparams`后缀)，默认为`None`.
- `-resume`: 模型的断续路径(文件夹目录)，默认为`None`.
- `-last_epoch`: 本次训练的起始轮次，默认为`None`.
- `-save_freq`: 保存checkpoint的轮次频率，默认为`1`.
- `-log_freq`: 日志输出打印的iter频率，默认为`100`.
- `-validate_freq`: 验证评估的轮次频率，默认为`10`.【验证时，会自动保存模型】
- `-accum_iter`: 梯度累加的指令，表示累加次数大小——默认为`1`，不累加.
- `-num_workers`: 数据加载的工作线程数，默认为`1`.
- `-ngpus`: 模型训练/验证时使用的gpu数，默认为`-1`.
- `-eval`: 启动评估，关闭训练.
- `-amp`: 启动混合精度训练.

> 倾斜字体部分为`main_single_gpu.py`以及`main_multi_gpu.py`中的必要命令行参数，其余参数根据需要自行配置修改即可。

## Visualization Attention Map
**(coming soon)**

## Reference
```
@misc{yang2021focal,
    title={Focal Self-attention for Local-Global Interactions in Vision Transformers}, 
    author={Jianwei Yang and Chunyuan Li and Pengchuan Zhang and Xiyang Dai and Bin Xiao and Lu Yuan and Jianfeng Gao},
    year={2021},
    eprint={2107.00641},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```