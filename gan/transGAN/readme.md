# TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up, [arxiv](https://arxiv.org/abs/2102.07074) 

PaddlePaddle training/validation code and pretrained models for **TransGAN**.

The official pytorch implementation is [here](https://github.com/VITA-Group/TransGAN).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT.git).


<p align="center">
<img src="./assets/TransGAN_1.png" alt="drawing" width="90%" height="90%"/>
<h4 align="center">TransGAN Model Overview</h4>
</p>







## Models Zoo
| Model                          | FID | Image Size |  Link        |
|--------------------------------|-----|------------|--------------|
| transgan_cifar10            |9.31 |32 |[google](https://drive.google.com/file/d/10NXjIUAkBmhPNiqTCYJ4hg3SWMw9BxCM/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16hi_kUZZOijNJNxocTiJXQ)(9vle)  |

## Notebooks
We provide a few notebooks in aistudio to help you get started:

**\*(coming soon)\***


## Requirements
- Python>=3.6
- yaml>=0.2.5
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0
- [yacs](https://github.com/rbgirshick/yacs)>=0.1.8

## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume the downloaded weight file is stored in `./transgan_cifar10.pdparams`, to use the `transgan_cifar10` model in python:
```python
from config import get_config
from models.ViT_custom import Generator
# config files in ./configs/
config = get_config('./configs/transgan_cifar10.yaml')
# build model
model = Generator(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./transgan_cifar10')
model.set_dict(model_state_dict)
```

## Generate Sample Images
To generate sample images from pretrained models, download the pretrained weights, and run the following script using command line:
```shell
sh run_generate.sh
```
or 
```shell
python generate.py \
  -cfg='./configs/transgan_cifar10.yaml' \
  -num_out_images=16 \
  -out_folder='./images_cifar10' \
  -pretrained='./transgan_cifar10.pdparams'
```
The output images are stored in `-out_folder` path.


## Evaluation
To evaluate TransGAN model performance on Cifar10 with a single GPU, run the following script using command line:
```shell
sh run_eval_cifar.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg="./configs/transgan_cifar10.yaml" \
  -dataset='cifar10' \
  -batch_size=32 \
  -eval \
  -pretrained='./transgan_cifar10'
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
    -cfg='./configs/transgan_cifar10.yaml' \
    -dataset='cifar10' \
    -batch_size=32 \
    -data_path='/dataset/imagenet' \
    -eval \
    -pretrained='./transgan_cifar10'
```

</details>



## Training
To train the TransGAN model on Cifar10 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
  -cfg="./configs/transgan_cifar10.yaml" \
  -dataset='cifar10' \
  -batch_size=32 \
```
<details>

<summary>
Run evaluation using multi-GPUs:
</summary>


```shell
sh run_train_multi.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main_multi_gpu.py \
    -cfg='./configs/transgan_cifar10.yaml' \
    -dataset='cifar10' \
    -batch_size=16 \
    -data_path='/dataset/imagenet' \
```

</details>



## Visualization of Generated Images
### Generated Images after Training
<img src="./assets/cifar_9_2.jpg" alt="drawing" width="40%" height="40%"/>
<figcaption align = "center">Generated Images from Cifar10 datasets</figcaption>

### Generated Images during Training 
**(coming soon)**

## Reference
```
@article{jiang2021transgan,
  title={Transgan: Two transformers can make one strong gan},
  author={Jiang, Yifan and Chang, Shiyu and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2102.07074},
  year={2021}
}
```
