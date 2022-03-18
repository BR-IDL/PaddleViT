# MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer, [arxiv](https://arxiv.org/abs/2110.02178) 

PaddlePaddle training/validation code and pretrained models for **MobileViT**.

The official apple implementation is [here](https://github.com/apple/ml-cvnets).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./mobilevit.png" alt="drawing" width="100%" height="100%"/>
    <h4 align="center">MobileViT Transformer Model Overview</h4>
</p>

### Update 
* Update (2022-03-16): Code is refactored.
* Update (2021-12-30): Add multi scale sampler DDP and update mobilevit_s model weights.
* Update (2021-11-02): Pretrained model weights (mobilevit_s) is released.
* Update (2021-11-02): Pretrained model weights (mobilevit_xs) is released.
* Update (2021-10-29): Pretrained model weights (mobilevit_xxs) is released.
* Update (2021-10-20): Initial code is released.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| mobilevit_xxs   				| 70.31| 89.68 | 1.32M   | 0.44G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1l3L-_TxS3QisRUIb8ohcv318vrnrHnWA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KFZ5G834_-XXN33W67k8eg?pwd=axpc) |
| mobilevit_xs   				| 74.47| 92.02 | 2.33M   | 0.95G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1oRMA4pNs2Ba0LYDbPufC842tO4OFcgwq/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1IP8S-S6ZAkiL0OEsiBWNkw?pwd=hfhm) |
| mobilevit_s   				| 76.74| 93.08 | 5.59M   | 1.88G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1ibkhsswGYWvZwIRjwfgNA4-Oo2stKi0m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1-rI6hiCHZaI7os2siFASNg?pwd=34bg) |
| mobilevit_s*  		    	| 77.83| 93.83 | 5.59M   | 1.88G   | 256        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1BztBJ5jzmqgDWfQk-FB_ywDWqyZYu2yG/view?usp=sharing)/[baidu](https://pan.baidu.com/s/19YepMAO-sveBOLA4aSjIEQ?pwd=92ic) |



> The results are evaluated on ImageNet2012 validation set.
> 
> All models are trained from scratch using **PaddleViT**.
>
> \* means model is trained from scratch using PaddleViT using multi scale sampler DDP.





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

For example, assume weight file is downloaded in `./mobilevit_s.pdparams`, to use the `mobilevit_s` model in python:
```python
from config import get_config
from mobilevit import build_mobilevit as build_model
# config files in ./configs/
config = get_config('./configs/mobilevit_s.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./mobilevit_s.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate MobileViT model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./mobilevit_s.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the MobileViT model on ImageNet2012, run the following script using command line:
```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.

## Finetuning
To finetune the MobileViT model on ImageNet2012, run the following script using command line:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/mobilevit_s.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./mobilevit_s.pdparams' \
-amp
```
> Note: use `-pretrained` argument to set the pretrained model path, you may also need to modify the hyperparams defined in config file.


## Reference
```
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```
