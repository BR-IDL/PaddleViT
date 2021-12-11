# Scaling Local Self-Attention for Parameter Efficient Visual Backbones, [arxiv](https://https://arxiv.org/abs/2103.12731) 

PaddlePaddle training/validation code and pretrained models for **HaloNet**.

The official pytorch implementation is N/A.

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./img1.png" alt="drawing" width="100%" height="100%"/>
    <h4 align="center">HaloNet local self-attention architecture</h4>
</p>

### Update 
* Update (2021-12-09): Initial code and ported weights are released.

## Models Zoo
| Model          | Acc@1 	| Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|----------------|----------|-------|---------|--------|------------|----------|---------------|--------------|
| halonet26t 	 | 79.10	| 94.31	| 12.5M    | 3.2G   | 256        | 0.95     | bicubic       |[google](https://drive.google.com/file/d/1F_a1brftXXnPM39c30NYe32La9YZQ0mW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FSlSTuYMpwPJpi4Yz2nCTA)(ednv)  |
| halonet50ts 	 | 81.65	| 95.61	| 22.8M    | 5.1G   | 256        | 0.94     | bicubic       |[google](https://drive.google.com/file/d/12t85kJcPA377XePw6smch--ELMBo6p0Y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1X4LM-sqoTKG7CrM5BNjcdA)(3j9e)  |

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

For example, assume the downloaded weight file is stored in `./halonet_50ts_256.pdparams`, to use the `halonet_50ts_256` model in python:
```python
from config import get_config
from halonet import build_halonet 
# config files in ./configs/
config = get_config('./configs/halonet_50ts_256.yaml')
# build model
model = build_model(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./halonet_50ts_256')
model.set_dict(model_state_dict)
```

## Evaluation
To evaluate HaloNet model performance on ImageNet2012 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/halonet_50ts_256.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./halonet_50ts_256'
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
    -cfg='./configs/halonet_50ts_256.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./halonet_50ts_256'
```

</details>


## Training
To train the MobileVit XXS model on ImageNet2012 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_singel_gpu.py \
  -cfg='./configs/halonet_50ts_256.yaml' \
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
    -cfg='./configs/halonet_50ts_256.yaml' \
    -dataset='imagenet2012' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>


## Visualization Attention Map
**(coming soon)**

## Reference
```
@inproceedings{vaswani2021scaling,
  title={Scaling local self-attention for parameter efficient visual backbones},
  author={Vaswani, Ashish and Ramachandran, Prajit and Srinivas, Aravind and Parmar, Niki and Hechtman, Blake and Shlens, Jonathon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12894--12904},
  year={2021}
}
```
