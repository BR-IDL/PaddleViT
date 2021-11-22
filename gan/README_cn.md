简体中文 | [English](./README.md)

# PaddleViT-GAN: GAN 领域的 Visual Transformer 模型
  
PaddlePaddle **GAN**的训练/评估代码以及预训练模型。

此实现是[PaddleViT](https://github.com/BR-IDL/PaddleViT)项目的一部分。

## 更新 
更新 (2021-08-25): 已上传初始化文件.

## 快速开始

 以下链接提供了每个模型架构的代码和详细用法：
1. **[Styleformer](./Styleformer)**
2. **[TransGAN](./transGAN)**


## 安装
该模块在 Python3.6+ 和 PaddlePaddle 2.1.0+ 上进行了测试，大多数依赖项通过PaddlePaddle安装，您只需要安装以下依赖项：

```shell
pip install yacs yaml lmdb
```
然后 下载 github repo:
```shell
git clone https://github.com/xperzy/PPViT.git
cd PPViT/image_classification
```

## 基本用法
### 数据准备
**Cifar10**, **STL10**, **Celeba** 和 **LSUNchurch** 数据集以如下结构使用:
#### [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html):
   
   我们使用 `paddle.io.Dataset.Cifar10` 创建Cifar10 dataset, 不需要手动的下载或准备数据.
#### [STL10](https://cs.stanford.edu/~acoates/stl10/):
```
│STL10/
├── train_X.bin
│── train_y.bin
├── test_X.bin
│── test_y.bin
│── unlabeled.bin
```
#### [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
```
│Celeba/
├──img_align_celeba/
│  ├── 000017.jpg
│  │── 000019.jpg
│  ├── 000026.jpg
│  │── ......
```
#### [LSUN-church](https://www.yf.io/p/lsun):
```
│LSUNchurch/
├──church_outdoor_train_lmdb/
│  ├── data.mdb
│  │── lock.mdb
```
### Demo 示例
对于具体模型示例，进入模型文件夹，下载预训练权重文件，例如`./cifar10.pdparams`, 然后在python中使用 `styleformer_cifar10` 模型:
```python
from config import get_config
from generator import Generator
# config files in ./configs/
config = get_config('./configs/styleformer_cifar10.yaml')
# build model
model = Generator(config)
# load pretrained weights, .pdparams is NOT needed
model_state_dict = paddle.load('./cifar10')
model.set_dict(model_state_dict)
```

### 生成示例图像
要想从预训练模型中生成示例图像，请先下载预训练权重，然后使用命令行运行以下脚本：
```shell
sh run_generate.sh
```
or 
```shell
python generate.py \
  -cfg='./configs/styleformer_cifar10.yaml' \
  -num_out_images=16 \
  -out_folder='./images_cifar10' \
  -pretrained='./cifar10.pdparams'
```
输出图像存储在 `-out_folder` 路径中.

> 注意：具体用法见各模型文件夹中的README文件.

## Basic Concepts
PaddleViT图像分类模块为每个模型在每一个单独文件夹中以相似结构进行开发，每个实现中大约有3种类型的类和2种类型的脚本：
1. **Model classes** 例如 **[ViT_custom.py](./transGAN/models/ViT_custom.py)**, 其中定义了核心 *transformer model* 以及相关方法.
   
2. **Dataset classes** 例如 **[dataset.py](./gan/transGAN/datasets.py)**, 其中定义了 dataset, dataloader, data transforms. 我们提供了自定义数据加载的实现方式，并且支持单GPU和多GPU加载.
   
3. **Config classes** 例如**[config.py](./gan/transGAN/config.py)**, 其中定义了模型训练/验证的配置. 通常，您不需要在配置文件中更改项目，可以通过python `arguments` 或者 `.yaml` 配置文件来更新配置. 您可以查看[here](../docs/ppvit-config.md) 了解关于配置设计和使用的详细信息.
   
4. **main scripts** 例如 **[main_single_gpu.py](./transGAN/main_single_gpu.py)**, 其中定义了整个训练/验证过程。提供了训练/验证的主要步骤，例如日志记录、加载/保存模型、微调等。 多-GPU的训练/验证过程在单独的python脚本 `main_multi_gpu.py`中实现.
   
5. **run scripts** 例如 **[run_eval_cifar.sh](./transGAN/run_eval_cifar.sh)**, 其中定义了用于运行使用特定配置和参数的python脚本的命令.
   

## Model Architectures

PaddleViT 目前支持以下 **transfomer based models**:
1. **[TransGAN](./transGAN)** (from Seoul National University and NUUA), released with paper [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up](https://arxiv.org/abs/2102.07074), by Yifan Jiang, Shiyu Chang, Zhangyang Wang.
2. **[Styleformer](./Styleformer)** (from Facebook and Sorbonne), released with paper [Styleformer: Transformer based Generative Adversarial Networks with Style Vector](https://arxiv.org/abs/2106.07023), by Jeeseung Park, Younggeun Kim.



## Contact
如果您有任何问题, 请在我们的Github上创建一个[issue](https://github.com/BR-IDL/PaddleViT/issues).
