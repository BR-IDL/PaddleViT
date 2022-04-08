# ConvMLP: Hierarchical Convolutional MLPs for Vision, [arxiv](https://arxiv.org/abs/2109.04454) 

PaddlePaddle training/validation code and pretrained models for **ConvMLP**.

The official and 3rd party pytorch implementation are [here](https://github.com/SHI-Labs/Convolutional-MLPs).


This implementation is developed by [PPViT](https://github.com/xperzy/PPViT/tree/master).

<p align="center">
<img src="./convmlp.png" alt="drawing" width="90%" height="90%"/>
<h4 align="center">ViP Model Overview</h4>
</p>



### Update 
Update (2022-04-08): Code is uploaded.
Update (2021-09-26): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| convmlp_s			  			| 76.76 | 93.40 | 9.0M    | 2.4G   |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1D8kWVfQxOyyktqDixaZoGXB3wVspzjlc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WseHYALFB4Of3Dajmlt45g?pwd=3jz3) |
| convmlp_m			  			| 79.03 | 94.53 | 17.4M   | 4.0G   |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1TqVlKHq-WRdT9KDoUpW3vNJTIRZvix_m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1koipCAffG6REUyLYk0rGAQ?pwd=vyp1) |
| convmlp_l			  			| 80.15 | 95.00 | 42.7M   | 10.0G  |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1KXxYogDh6lD3QGRtFBoX5agfz81RDN3l/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1f1aEeVoySzImI89gkjcaOA?pwd=ne5x) |


> *The results are evaluated on ImageNet2012 validation set.
> 
> Note: ConvMLP weights are ported from [here](https://github.com/SHI-Labs/Convolutional-MLPs)

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

For example, assume weight file is downloaded in `./convmlp_s.pdparams`, to use the `convmlp_s` model in python:
```python
from config import get_config
from convmlp import build_convmlp as build_model
# config files in ./configs/
config = get_config('./configs/convmlp_s.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./convmlp_s.pdparams')
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
-cfg='./configs/convmlp_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./convmlp_s.pdparams' \
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
-cfg='./configs/convmlp_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.




## Reference
```
@article{li2021convmlp,
      title={ConvMLP: Hierarchical Convolutional MLPs for Vision}, 
      author={Jiachen Li and Ali Hassani and Steven Walton and Humphrey Shi},
      year={2021},
      eprint={2109.04454},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
