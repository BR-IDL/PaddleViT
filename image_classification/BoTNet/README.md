# Bottleneck Transformers for Visual Recognition, [arxiv](https://arxiv.org/abs/2101.11605) 

PaddlePaddle training/validation code and pretrained models for **BoTNet**.

The official pytorch implementation is N/A. The 3rd party timm pytorch implementation is [here](rwightman/pytorch-image-models)

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./img.png" alt="drawing" width="60%" height="60%"/>
    <h4 align="center">BotNet architecture</h4>
</p>

### Update 
* Update (2022-04-11): Code is updated.
* Update (2021-12-22): Initial code and ported weights are released.

## Models Zoo
| Model          | Acc@1 	| Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|----------------|----------|-------|---------|--------|------------|----------|---------------|--------------|
| botnet50 	 | 77.38	| 93.56	| 20.9M    | 5.3G   | 224        | 0.875     | bicubic       |[google](https://drive.google.com/file/d/1S4nxgRkElT3K4lMx2JclPevmP3YUHNLw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1CW40ShBJQYeFgdBIZZLSjg?pwd=wh13)  |


> *The results are evaluated on ImageNet2012 validation set.


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
- `train_list.txt`: list of relative paths and labels of training images. You can download it from: [google](https://drive.google.com/file/d/10YGzx_aO3IYjBOhInKT_gY6p0mC3beaC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1G5xYPczfs9koDb7rM4c0lA?pwd=a4vm?pwd=a4vm)
- `val_list.txt`: list of relative paths and labels of validation images. You can download it from: [google](https://drive.google.com/file/d/1aXHu0svock6MJSur4-FKjW0nyjiJaWHE/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1TFGda7uBZjR7g-A6YjQo-g?pwd=kdga?pwd=kdga) 


## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume weight file is downloaded in `./botnet50.pdparams`, to use the `botnet50` model in python:
```python
from config import get_config
from botnet import build_botnet50 as build_model
# config files in ./configs/
config = get_config('./configs/botnet50.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./botnet50.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/botnet50.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./botnet50.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the model on ImageNet2012, run the following script using command line:
```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/botnet50.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.



## Reference
```
@inproceedings{srinivas2021bottleneck,
  title={Bottleneck transformers for visual recognition},
  author={Srinivas, Aravind and Lin, Tsung-Yi and Parmar, Niki and Shlens, Jonathon and Abbeel, Pieter and Vaswani, Ashish},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16519--16529},
  year={2021}
}
```
