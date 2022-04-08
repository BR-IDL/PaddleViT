# CycleMLP: A MLP-like Architecture for Dense Prediction, [arxiv](https://arxiv.org/abs/2107.10224)

PaddlePaddle training/validation code and pretrained models for **CycleMLP**.

The official and 3rd party pytorch implementation are [here](https://github.com/ShoufaChen/CycleMLP).


This implementation is developed by [PPViT](https://github.com/BR-IDL/PaddleViT).

<p align="center">
<img src="./cyclemlp.png" alt="drawing" width="60%" height="60%"/>
<h4 align="center">CycleMLP Model Overview</h4>
</p>



### Update 
Update (2022-04-08): Code is updated.
Update (2021-09-24): Code is released and ported weights are uploaded.

## Models Zoo
| Model       | Acc@1 | Acc@5 | #Params | Image Size | Crop_pct | Interpolation | Link                                                         |
| ----------- | ----- | ----- | ------- | ---------- | -------- | ------------- | ------------------------------------------------------------ |
| cyclemlp_b1 | 78.85 | 94.60 | 15.1M   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/10WQenRy9lfOJF4xEHc9Mekp4zHRh0mJ_/view?usp=sharing)/[baidu](https://pan.baidu.com/s/11UQp1RkWBsZFOqit_uU80w?pwd=mnbr) |
| cyclemlp_b2 | 81.58 | 95.81 | 26.8M   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1dtQHCwtxNh9jgiHivN5iYpHe7uKRUjhk/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Js-Oq5vyiB7oPagn43cn3Q?pwd=jwj9) |
| cyclemlp_b3 | 82.42 | 96.07 | 38.3M   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/11kMq112tAwVE5llJIepIIixz74AjaJhU/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1b7cau1yPxqATA8X7t2DXkw?pwd=v2fy) |
| cyclemlp_b4 | 82.96 | 96.33 | 51.8M   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1vwJ0eD9Ic-NvLvCz1zEAmn7RxBMtd_v2/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1P3TlnXRFGWj9nVP5xBGGWQ?pwd=fnqd) |
| cyclemlp_b5 | 83.25 | 96.44 | 75.7M   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/12_I4cfOBfp7kC0RvmnMXFqrSxww6plRW/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-Cka1tNqGUQutkAP3VZXzQ?pwd=s55c) |







> *The results are evaluated on ImageNet2012 validation set.
> 
> Note: CycleMLP weights are ported from [here](https://github.com/ShoufaChen/CycleMLP)



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

For example, assume weight file is downloaded in `./cyclemlp_b1.pdparams`, to use the `cyclemlp_b1` model in python:
```python
from config import get_config
from cyclemlp import build_cyclemlp as build_model
# config files in ./configs/
config = get_config('./configs/cyclemlp_b1.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./cyclemlp_b1.pdparams')
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
-cfg='./configs/cyclemlp_b1.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./cyclemlp_b1.pdparams' \
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
-cfg='./configs/cyclemlp_b1.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.



## Reference
```
@article{chen2021cyclemlp,
  title={CycleMLP: A MLP-like Architecture for Dense Prediction},
  author={Chen, Shoufa and Xie, Enze and Ge, Chongjian and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2107.10224},
  year={2021}
}
```
