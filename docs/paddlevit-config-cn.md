## PaddleViT: 如何使用 config ?
> 示例代码: [here](../image_classification/ViT/config.py)

本文档介绍了**PaddleViT** 项目中使用`config` 的基础知识。

PPViT `config`中使用的核心模块是[yacs](https://github.com/rbgirshick/yacs) (0.1.8+). 与其他项目相似，PPViT `config`支持从[yaml](https://yaml.org/)文件加载，并且可以使用python [ArgumentParser](https://docs.python.org/3/library/argparse.html)进行配置。

> `yacs` 的完整用法可以在https://github.com/rbgirshick/yacs 中找到

### 1. 安装
#### 1.1 通过 `pip`安装
安装 `yacs` 版本 `0.1.8`:
```shell
$ pip install yacs==0.1.8
```
#### 1.2 从源码安装
你也可以从github下载 `yacs` :
```shell
$ git clone https://github.com/rbgirshick/yacs.git
$ cd yacs
$ python setup.py install
```

### 2. 基本概念和用法
#### 1. CfgNode
`CfgNode` 表示配置树中的一个内部节点，它是一个类似`dict`的容器，允许基于属性对其键进行访问。
```python
from yacs.config import CfgNode as CN

_C = CN()

_C.NUM_GPUS = 4
_C.NUM_WORKERS = 8
_C.BATCH_SIZE = 128

def get_config():
    return _C.clone()
```
#### 2. 使用 `merge_from_file()`读取 `.yaml`
`yacs`允许读取YAML文件来覆盖 `CfgNode`. 您可以为每一个实验创建一个`.yaml` 文件，它只会更改实验中的选项。

YAML文件的一些基本格式:
```YAML
key:    # YAML uses 'key: value' paris, separated using ':'
    child_key: value    # indent can be used to show different levels
    child_KEY2: value3  # YAML is case sensitive
    c_arr: [val1, val2, val3]   # array can be used in value
    c_bool: True    # True/true/TRUE are all OK
    c_float: 3.1415 # float is allowed
    c_string: no quote is allowed # "", '', no quote are all OK
```

`merge_from_file()` 可用于覆盖当前 `CfgNode`:
```python
cfg = get_config()
cfg.merge_from_file('experiment_1.yaml')
print(cfg)
```

#### 3. 通过 `ArgumentParser`更新配置
您可以使用python `ArgumentParser` 编写您自己的方法来更新配置，例如：
```python
def update_config(config, args)
    if args.cfg:    # update from .yaml file
        upate_config_from_file(config, args.cfg)
    if args.batch_size: # update BATCH_SIZE
        config.BATCH_SIZE = args.batch_size
    if args.eval:
        config.EVAL = True
    return config
```




### 4. PPViT配置的使用指南:
#### STEP 1: 创建 config.py
创建一个python文件config.py， 用于定义 **所有配置选项**。 它应该为所有选项提供合适的默认值并记录下来。
通常，`config.py`应该有：
- `DATA`: 定义数据集路径、输入图像尺寸和batch_size等。
- `MODEL`:
    - 模型的常规选项，例如模型名称，类别数量等。
    - `TRANS`: transformer的相关选项，例如mlp维度，hidden维度，heads数量等。
- `TRAIN`: 与训练相关的选项，例如epochs, lr, weight decay等。

在`config.py`中，你应该实现`update_config(config, args)`，它是从`ArgumentParser`中读取当前的 `config` 和 `args`以使用命令行选项更新配置。

#### STEP 2: 
在你的`main.py`中，创建`ArgumentParser`，它包含`config.py`中`update_config(config, args)` 方法中的所有选项，例如：

```python
  parser = argparse.ArgumentParser('ViT')
  parser.add_argument('-cfg', type=str, default=None)
  parser.add_argument('-dataset', type=str, default=None)
  parser.add_argument('-batch_size', type=int, default=None)
  parser.add_argument('-image_size', type=int, default=None)
  parser.add_argument('-data_path', type=str, default=None)
  parser.add_argument('-ngpus', type=int, default=None)
  parser.add_argument('-pretrained', type=str, default=None)
  parser.add_argument('-eval', action='store_true')
  args = parser.parse_args()

  # get default config
  config = get_config()
  # update config by arguments
  config = update_config(config, args)
```

然后，您可以使用基于属性的访问来获取配置选项值。

#### STEP 3:
你应该为每个实验创建一个单独的`.yaml` 文件，例如：
```yaml
DATA:
    IMAGE_SIZE: 224
    CROP_PCT: 0.875
MODEL:
    TYPE: ViT
    NAME: vit_large_patch16_224
    TRANS:
    PATCH_SIZE: 16
    HIDDEN_SIZE: 1024
    MLP_DIM: 4096 # same as mlp_ratio = 4.0
    NUM_LAYERS: 24
    NUM_HEADS: 16
    QKV_BIAS: True
```

如果你将命令行参数`-cfg`设置为`.yaml` 文件路径，配置将被文件选项覆盖。
> 注意：`.yaml`覆盖了 `args`之前的配置，因此`args`中的选项是当前选项。
