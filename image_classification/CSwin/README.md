# CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows, [arxiv](https://arxiv.org/pdf/2107.00652.pdf) 

PaddlePaddle training/validation code and pretrained models for **CSWin Transformer**.

The official pytorch implementation is [here](https://github.com/microsoft/CSWin-Transformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">

<img src="./cswin1.png" alt="drawing" width="90%" height="90%"/>
<img src="./cswin2.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center">CSWin Transformer Model Overview</h4>
</p>


### Update 
- Update (2022-03-16): Code is refactored.
- Update (2021-09-27): Model FLOPs and # params are uploaded.
- Update (2021-08-11): Code is released and ported weights are uploaded.

## Models Zoo
| Model                         | Acc@1 | Acc@5 | #Params | FLOPs  | Image Size | Crop_pct | Interpolation | Link         |
|-------------------------------|-------|-------|---------|--------|------------|----------|---------------|--------------|
| cswin_tiny_224  				| 82.81 | 96.30 | 22.3M   | 4.2G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1l-JY0u7NGyD6SjkyiyNnDx3wFFT1nAYO/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1L5FqU7ImWAhQHAlSilqVAw?pwd=4q3h) |
| cswin_small_224 				| 83.60 | 96.58 | 34.6M   | 6.5G   | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/10eEBk3wvJdQ8Dy58LvQ11Wk1K2UfPy-E/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1FiaNiWyAuWu1IBsUFLUaAw?pwd=gt1a) |
| cswin_base_224  				| 84.23 | 96.91 | 77.4M   | 14.6G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1YufKh3DKol4-HrF-I22uiorXSZDIXJmZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1koy8hXyGwvgAfUxdlkWofg?pwd=wj8p) |
| cswin_base_384  				| 85.51 | 97.48 | 77.4M   | 43.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1qCaFItzFoTYBo-4UbGzL6M5qVDGmJt4y/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1WNkY7o_vP9KJ8cd5c7n2sQ?pwd=rkf5) |
| cswin_large_224 				| 86.52 | 97.99 | 173.3M  | 32.5G  | 224        | 0.9      | bicubic       | [google](https://drive.google.com/file/d/1V1hteGK27t1nI84Ac7jdWfydBLLo7Fxt/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1KgIX6btML6kPiPGkIzvyVA?pwd=b5fs) |
| cswin_large_384 				| 87.49 | 98.35 | 173.3M  | 96.1G  | 384        | 1.0      | bicubic       | [google](https://drive.google.com/file/d/1LRN_6qUz71yP-OAOpN4Lscb8fkUytMic/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1eCIpegPj1HIbJccPMaAsew?pwd=6235) | 

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

For example, assume weight file is downloaded in `./cswin_tiny_224.pdparams`, to use the `cswin_tiny_224` model in python:
```python
from config import get_config
from cswin import build_cswin as build_model
# config files in ./configs/
config = get_config('./configs/cswin_tiny_224.yaml')
# build model
model = build_model(config)
# load pretrained weights
model_state_dict = paddle.load('./cswin_tiny_224.pdparams')
model.set_state_dict(model_state_dict)
```

## Evaluation
To evaluate CSwin model performance on ImageNet2012, run the following script using command line:
```shell
sh run_eval_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/cswin_tiny_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-eval \
-pretrained='./cswin_tiny_224.pdparams' \
-amp
```
> Note: if you have only 1 GPU, change device number to `CUDA_VISIBLE_DEVICES=0` would run the evaluation on single GPU.


## Training
To train the CSwin model on ImageNet2012, run the following script using command line:
```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/cswin_tiny_224.yaml' \
-dataset='imagenet2012' \
-batch_size=256 \
-data_path='/dataset/imagenet' \
-amp
```
> Note: it is highly recommanded to run the training using multiple GPUs / multi-node GPUs.

## Finetuning
To finetune the Swin model on ImageNet2012, run the following script using command line:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_multi_gpu.py \
-cfg='./configs/cswin_base_384.yaml' \
-dataset='imagenet2012' \
-batch_size=16 \
-data_path='/dataset/imagenet' \
-pretrained='./cswin_base_224.pdparams' \
-amp
```
> Note: use `-pretrained` argument to set the pretrained model path, you may also need to modify the hyperparams defined in config file.



## Reference
```
@article{dong2021cswin,
  title={CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows},
  author={Dong, Xiaoyi and Bao, Jianmin and Chen, Dongdong and Zhang, Weiming and Yu, Nenghai and Yuan, Lu and Chen, Dong and Guo, Baining},
  journal={arXiv preprint arXiv:2107.00652},
  year={2021}
}
```