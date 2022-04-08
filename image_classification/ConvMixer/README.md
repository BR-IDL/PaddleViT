# ConvMixer: Patches Are All You Need? [arxiv](https://arxiv.org/abs/2201.09792)

PaddlePaddle training/validation code and pretrained models for **ConvMixer**.

The official pytorch implementation is [here](https://github.com/tmp-iclr/convmixer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).

<p align="center">
<img src="./convmixer.png" alt="drawing" width="90%" height="90%"/>
<h4 align="center">ConvMixer Model Overview</h4>
</p>



### Update 
- Update (2022-04-08): Code is updated.
- Update (2021-11-04): Model weights are updated.
- Update (2021-10-13): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| convmixer_1024_20  			| 76.94 | 93.35 | 24.5M   | 9.5G   |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/1R7zUSl6_6NFFdNOe8tTfoR9VYQtGfD7F/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1DgGA3qYu4deH4woAkvjaBw?pwd=qpn9) |
| convmixer_768_32  			| 80.16 | 95.08 | 21.2M   | 20.8G  |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/196Lg_Eet-hRj733BYASj22g51wdyaW2a/view?usp=sharing)/[baidu](https://pan.baidu.com/s/17CbRNzY2Sy_Cu7cxNAkWmQ?pwd=m5s5) |
| convmixer_1536_20  			| 81.37 | 95.62 | 51.8M   | 72.4G  |    224     | 0.96     | bicubic       | [google](https://drive.google.com/file/d/1-LlAlADiu0SXDQmE34GN2GBhqI-RYRqO/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1R-gSzhzQNfkuZVxsaE4vEw?pwd=xqty) |
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

For example, assume weight file is downloaded in `./convmixer_768_32.pdparams`, to use the `convmixer_768_32` model in python:
```python
from config import get_config
from convmixer import build_convmixer as build_model
# config files in ./configs/
config = get_config('./configs/convmixer_768_32.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./convmixer_768_32.pdparams')
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
-cfg='./configs/convmixer_768_32.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./convmixer_768_32.pdparams' \
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
-cfg='./configs/convmixer_768_32.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.


## Reference
```
@article{trockman2022patches,
  title={Patches Are All You Need?},
  author={Trockman, Asher and Kolter, J Zico},
  journal={arXiv preprint arXiv:2201.09792},
  year={2022}
}
```


