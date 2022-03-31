# Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition, [arxiv](https://arxiv.org/abs/2106.12368) 

PaddlePaddle training/validation code and pretrained models for **ViP**.

The official and 3rd party pytorch implementation are [here](https://github.com/Andrew-Qibin/VisionPermutator).


This implementation is developed by [PPViT](https://github.com/BR-IDL/PaddleViT/).

<p align="center">
<img src="./vip_1.png" alt="drawing" width="90%" height="90%"/>
<img src="./vip_2.png" alt="drawing" width="90%" height="90%"/>
<h4 align="center">ViP Model Overview</h4>
</p>


### Update 
- Update (2022-03-30): Code is refactored.
- Update (2021-11-03): Code and weights are updated.
- Update (2021-09-23): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| vip_s7  						| 81.48 | 95.76 | 25.1M   | 7.0G   |    224     | 0.9    | bicubic       | [google](https://drive.google.com/file/d/16bZkqzbnN08_o15k3MzbegK8SBwfQAHF/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1uY0FsNPYaM8cr3ZCdAoVkQ?pwd=mh9b) |
| vip_m7  						| 82.64 | 96.12 | 55.3M   | 16.4G  |    224     | 0.9    | bicubic       | [google](https://drive.google.com/file/d/11lvT2OXW0CVGPZdF9dNjY_uaEIMYrmNu/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1j3V0Q40iSqOY15bTKlFFRw?pwd=hvm8) |
| vip_l7  						| 83.18 | 96.37 | 87.8M   | 24.5G  |    224     | 0.875    | bicubic       | [google](https://drive.google.com/file/d/1bK08JorLPMjYUep_TnFPKGs0e1j0UBKJ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1I5hnv3wHWEaG3vpDqaNL-w?pwd=tjvh) |
> *The results are evaluated on ImageNet2012 validation set.
> 
> Note: ViP weights are ported from [here](https://github.com/Andrew-Qibin/VisionPermutator)


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

For example, assume weight file is downloaded in `./vip_s.pdparams`, to use the `vip_s` model in python:
```python
from config import get_config
from vip import build_vip as build_model
# config files in ./configs/
config = get_config('./configs/vip_s.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./vip_s.pdparams')
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
-cfg='./configs/vip_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./vip_s.pdparams' \
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
-cfg='./configs/vip_s.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.


## Reference
```
@misc{hou2021vision,
    title={Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
    author={Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
    year={2021},
    eprint={2106.12368},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
