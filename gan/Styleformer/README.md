# Styleformer: Transformer based Generative Adversarial Networks with Style Vector, [arxiv](https://arxiv.org/abs/2106.07023v2) 

PaddlePaddle training/validation code and pretrained models for **Styleformer**.

The official pytorch implementation is [here](https://github.com/Jeeseung-Park/Styleformer).

This implementation is developed by [PaddleViT](https://github.com/BR-IDL/PaddleViT).



<img src="./Styleformer.jpg" alt="drawing" width="100%" height="100%"/>
<figcaption align = "center">Styleformer Model Overview</figcaption>

### Update 
Update (2021-08-17): Code is released and ported weights are uploaded.

## Models Zoo
| Model                          | FID | Image Size | Crop_pct | Interpolation | Model        |
|--------------------------------|-----|------------|----------|---------------|--------------|
| styleformer_cifar10            |2.73 | 32         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1iW76QmwbYz6GeAPQn8vKvsG0GvFdhV4T/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1Ax7BNEr1T19vgVjXG3rW7g)(ztky)  |
| styleformer_stl10              |15.65| 48         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/15p785y9eP1TeoqUcHPbwFPh98WNof7nw/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1rSORxMYAiGkLQZ4zTA2jcg)(i973)|
| styleformer_celeba             |3.32 | 64         | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1_YauwZN1osvINCboVk2VJMscrf-8KlQc/view?usp=sharing)/[baidu](https://pan.baidu.com/s/16NetcPxLQF9C_Zlp1SpkLw)(fh5s) |
| styleformer_lsun               | 9.68 | 128        | 1.0      | lanczos       |[google](https://drive.google.com/file/d/1i5kNzWK04ippFSmrmcAPMItkO0OFukTd/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1jTS9ExAMz5H2lhue4NMV2A)(158t)|
> *The results are evaluated on Cifar10, STL10, Celeba and LSUNchurch dataset, using **fid50k_full** metric.
## Notebooks
We provide a few notebooks in aistudio to help you get started:

**\*(coming soon)\***


## Requirements
- Python>=3.6
- yaml>=0.2.5
- lmdb>=1.2.1
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)>=2.1.0
- [yacs](https://github.com/rbgirshick/yacs)>=0.1.8

## Data 
STL10, Celeba and LSUNchurch dataset is used in the following folder structure:
```
│STL10/
├── train_X.bin
│── train_y.bin
├── test_X.bin
│── test_y.bin
│── unlabeled.bin
```
```
│Celeba/
├──img_align_celeba/
│  ├── 000017.jpg
│  │── 000019.jpg
│  ├── 000026.jpg
│  │── unlabeled.bin
│  │── ......
```
```
│LSUNchurch/
├──church_outdoor_train_lmdb/
│  ├── data.mdb
│  │── lock.mdb
```

## Usage
To use the model with pretrained weights, download the `.pdparam` weight file and change related file paths in the following python scripts. The model config files are located in `./configs/`.

For example, assume the downloaded weight file is stored in `./cifar10.pdparams`, to use the `styleformer_cifar10` model in python:
```python
from config import get_config
from generator import Generator
# config files in ./configs/
config = get_config('./configs/styleformer_cifar10.yaml')
# build model
model = Generator(config)
# load pretrained weights
model_state_dict = paddle.load('./cifar10.pdparams')
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
  -cfg=./configs/styleformer_cifar10.yaml \
  -num_out_images=16 \
  -out_folder=./images_cifar10 \
  -pretrained=/path/to/pretrained/model/cifar10  # .pdparams is NOT needed
```
The output images are stored in `-out_folder` path.


## Evaluation
To evaluate Styleformer model performance on Cifar10 with a single GPU, run the following script using command line:
```shell
sh run_eval.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg=./configs/styleformer_cifar10.yaml \
    -dataset=cifar10 \
    -batch_size=32 \
    -eval \
    -pretrained=/path/to/pretrained/model/cifar10 # .pdparams is NOT needed
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
python main_single_gpu.py \
    -cfg=./configs/styleformer_cifar10.yaml \
    -dataset=cifar10 \
    -batch_size=32 \
    -eval \
    -pretrained=/path/to/pretrained/model/cifar10.pdparams  # .pdparams is NOT needed
```

</details>


## Training
To train the Styleformer Transformer model on Cifar10 with single GPU, run the following script using command line:
```shell
sh run_train.sh
```
or
```shell
CUDA_VISIBLE_DEVICES=0 \
python main_single_gpu.py \
    -cfg='./configs/styleformer_cifar10.yaml' \
    -dataset=cifar10 \
    -batch_size=32 \
    -pretrained=/path/to/pretrained/model/cifar10 # .pdparams is NOT needed
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_single_gpu.py \
    -cfg=./configs/styleformer_cifar10.yaml \
    -dataset=cifar10 \
    -batch_size=32 \
    -pretrained=/path/to/pretrained/model/cifar10  # .pdparams is NOT needed
```

</details>


## Visualization of Generated Images
### Generated Images after Training
<img src="./fig2.png" alt="drawing" width="60%" height="60%"/>
<figcaption align = "center">Generated Images from CelebA(left) and LSUN-church(right) datasets</figcaption>

### Generated Images during Training 
**(coming soon)**

## Reference
```
@article{park2021styleformer,
      title={Styleformer: Transformer based Generative Adversarial Networks with Style Vector}, 
      author={Jeeseung Park and Younggeun Kim},
      year={2021},
      eprint={2106.07023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
