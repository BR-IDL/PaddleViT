English | [简体中文](./paddlevit-config-cn.md)

## PaddleViT: How to use config?
> sample code: [here](../image_classification/ViT/config.py)

This document presents the basics of `config` that used in **PaddleViT** project. 

The core module used in PPViT `config` is [yacs](https://github.com/rbgirshick/yacs) (0.1.8+). Similar as other projects, PPViT `config` supports loading from [yaml](https://yaml.org/) file, and configarable using python [ArgumentParser](https://docs.python.org/3/library/argparse.html).

> Full usage of `yacs` can be found in https://github.com/rbgirshick/yacs

### 1. Installation
#### 1.1 Install by `pip`
To intall `yacs` version `0.1.8`:
```shell
$ pip install yacs==0.1.8
```
#### 1.2 Install from source
You can also install `yacs` from gitub:
```shell
$ git clone https://github.com/rbgirshick/yacs.git
$ cd yacs
$ python setup.py install
```

### 2. Basic Concepts and Usage
#### 1. CfgNode
`CfgNode` represents an internal node in the configuration tree. It is a `dict`-like container, which allows attribute-based access to its keys.
```python
from yacs.config import CfgNode as CN

_C = CN()

_C.NUM_GPUS = 4
_C.NUM_WORKERS = 8
_C.BATCH_SIZE = 128

def get_config():
    return _C.clone()
```
#### 2. Read `.yaml` file using `merge_from_file()`
`yacs` allows reading YAML file to override `CfgNode`. You can create a `.yaml` file for each experiment, which will only change the options in that expriment.

Some basic format in YAML file:
```YAML
key:    # YAML uses 'key: value' paris, separated using ':'
    child_key: value    # indent can be used to show different levels
    child_KEY2: value3  # YAML is case sensitive
    c_arr: [val1, val2, val3]   # array can be used in value
    c_bool: True    # True/true/TRUE are all OK
    c_float: 3.1415 # float is allowed
    c_string: no quote is allowed # "", '', no quote are all OK
```

`merge_from_file()` can be used to override current `CfgNode`:
```python
cfg = get_config()
cfg.merge_from_file('experiment_1.yaml')
print(cfg)
```

#### 3. Override by `ArgumentParser`
You can write your own method to update config using python `ArgumentParser`, e.g.:
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




### 4. Practical Guide of using config for PPViT:
#### STEP 1: Create config.py
Create a python file config.py, which is the place to define **all the configurable options**. It should be well documented and provide suitable default values for all options. 
Typically, `config.py` should have:
- `DATA`: defines the dataset path, input image size, and batch_size, etc.
- `MODEL`:
    - General options for your model, such as model name, num_classes, etc.
    - `TRANS`: transformer related options, such as mlp dimention, hidden dimension, number of heads, etc.
- `TRAIN`: training related options, such as num of epochs, lr, weight decay, etc.

In `config.py`, you should implement `update_config(config, args)`, which reads current `config` and `args` from `ArgumentParser` to update the config using commandline options.

#### STEP 2: 
In your `main.py`, create `ArgumentParser` which includes all the options in `update_config(config, args)` method from `config.py`, e.g.:
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

Then you can use the attribute-based access to get the config option values.

#### STEP 3:
You should create a single `.yaml` file for each experiment, e.g.,
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

If you set the command line argument `-cfg` to the `.yaml` file path, the config will be override with the file options. 
> **Note:** the `.yaml` overrides the config before the `args`, so the options in `args` are the current options.
