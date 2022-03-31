# ResMLP: Feedforward networks for image classification with data-efficient training, [arxiv](https://arxiv.org/abs/2105.03404) 

PaddlePaddle training/validation code and pretrained models for **ResMLP**.

The official and 3rd party pytorch implementation are [here](https://github.com/facebookresearch/deit) and [here](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py).


This implementation is developed by [PPViT](https://github.com/xperzy/PPViT/tree/master).

<p align="center">
<img src="./resmlp.png" alt="drawing" width="100%" height="100%"/>
<h4 align="center">ResMLP Model Overview</h4>
</p>

### Update 

- Update (2022-03-31): Code is refactored.
- Update (2020-09-27): Model FLOPs and # params are uploaded.
- Update (2020-09-24): Update new ResMLP weights.
- Update (2020-09-23): Add new ResMLP weights.
- Update (2020-08-11): Code is released and ported weights are uploaded.

## Models Zoo

**Original**:

| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| resmlp_24_224                	| 79.38 | 94.55 | 30.0M   | 6.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/15A5q1XSXBz-y1AcXhy_XaDymLLj2s2Tn/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1nLAvyG53REdwYNCLmp4yBA?pwd=jdcx) |
| resmlp_36_224             	| 79.77 | 94.89 | 44.7M   | 9.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1WrhVm-7EKnLmPU18Xm0C7uIqrg-RwqZL/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1QD4EWmM9b2u1r8LsnV6rUA?pwd=33w3) |
| resmlp_big_24_224         	| 81.04 | 95.02 | 129.1M  | 100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1KLlFuzYb17tC5Mmue3dfyr2L_q4xHTZi/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1oXU6CR0z7O0XNwu_UdZv_w?pwd=r9kb) |
| resmlp_12_distilled_224 		| 77.95 | 93.56 | 15.3M   |	3.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1cDMpAtCB0pPv6F-VUwvgwAaYtmP8IfRw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/15kJeZ_V1MMjTX9f1DBCgnw?pwd=ghyp) |
| resmlp_24_distilled_224 		| 80.76 | 95.22 | 30.0M   |	6.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/15d892ExqR1sIAjEn-cWGlljX54C3vihA/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1NgQtSwuAwsVVOB8U6N4Aqw?pwd=sxnx) |
| resmlp_36_distilled_224 		| 81.15 | 95.48 | 44.7M	  | 9.0G   | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1Laqz1oDg-kPh6eb6bekQqnE0m-JXeiep/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1p1xGOJbMzH_RWEj36ruQiw?pwd=vt85) |
| resmlp_big_24_distilled_224 	| 83.59 | 96.65 | 129.1M  |	100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/199q0MN_BlQh9-HbB28RdxHj1ApMTHow-/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1yUrfbqW8vLODDiRV5WWkhQ?pwd=4jk5) |
| resmlp_big_24_22k_224   		| 84.40 | 97.11 | 129.1M  | 100.7G | 224        | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1zATKq1ruAI_kX49iqJOl-qomjm9il1LC/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1VrnRMbzzZBmLiR45YwICmA?pwd=ve7i) |



> *The results are evaluated on ImageNet2012 validation set.
>
> Note: ResMLP weights are ported from [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mlp_mixer.py) and [facebookresearch](https://github.com/facebookresearch/deit/blob/main/README_resmlp.md)

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

For example, assume weight file is downloaded in `./resmlp_12_224.pdparams`, to use the `resmlp_12_224` model in python:
```python
from config import get_config
from resmlp import build_resmlp as build_model
# config files in ./configs/
config = get_config('./configs/resmlp_12_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./resmlp_12_224.pdparams')
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
-cfg='./configs/resmlp_12_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./resmlp_12_224.pdparams' \
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
-cfg='./configs/resmlp_12_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.


## Reference
```
@article{touvron2021resmlp,
  title={Resmlp: Feedforward networks for image classification with data-efficient training},
  author={Touvron, Hugo and Bojanowski, Piotr and Caron, Mathilde and Cord, Matthieu and El-Nouby, Alaaeldin and Grave, Edouard and Joulin, Armand and Synnaeve, Gabriel and Verbeek, Jakob and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2105.03404},
  year={2021}
}
```
