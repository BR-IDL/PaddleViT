# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration
Configuration for data, model archtecture, and training, etc.
Config can be set by .yaml file or by argparser(limited usage)
"""
import os
from yacs.config import CfgNode as CN
import yaml

_C = CN()
_C.BASE = ['']

# data settings - is ok
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 256 # train batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = None # val batch_size for single GPU
_C.DATA.DATA_PATH = '/dataset/imagenet/' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size: 224 for pretrain, 384 for finetune
_C.DATA.IMAGE_CHANNELS = 3  # input image channels: e.g., 3
_C.DATA.CROP_PCT = 0.875 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 2 # number of data loading threads
_C.DATA.IMAGENET_MEAN = [0.485, 0.456, 0.406] # [0.5, 0.5, 0.5]
_C.DATA.IMAGENET_STD = [0.229, 0.224, 0.225] # [0.5, 0.5, 0.5]

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'MobileFormer'
_C.MODEL.NAME = 'MobileFormer_26M'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.1
_C.MODEL.DROPPATH = 0.1
_C.MODEL.ATTENTION_DROPOUT = 0.1
_C.MODEL.MLP_DROPOUT = 0.1

# mobileformer architecture settings
_C.MODEL.MF = CN()
_C.MODEL.MF.IN_CHANNELS = 3
_C.MODEL.MF.TOKENS = [3, 128] # token size
_C.MODEL.MF.NUM_HEAD = 4
_C.MODEL.MF.MLP_RATIO= 2.0
_C.MODEL.MF.ALPHA = 1.0
_C.MODEL.MF.QKV_BIAS = True
_C.MODEL.MF.POINTWISECONV_GROUPS = 4 # the groups of pointwise 1x1conv

# mobileformer architecture settings -- dyrelu
_C.MODEL.MF.DYRELU =  CN()
_C.MODEL.MF.DYRELU.USE_DYRELU = True
_C.MODEL.MF.DYRELU.REDUCE = 6.0
_C.MODEL.MF.DYRELU.DYRELU_K = 2
_C.MODEL.MF.DYRELU.COEFS = [1.0, 0.5]
_C.MODEL.MF.DYRELU.CONSTS = [1.0, 0.0]

# mobileformer architecture settings -- stem
_C.MODEL.MF.STEM =  CN()
_C.MODEL.MF.STEM.OUT_CHANNELS = 8
_C.MODEL.MF.STEM.KERNELS = 3
_C.MODEL.MF.STEM.STRIEDS = 2
_C.MODEL.MF.STEM.PADDINGS = 1

# mobileformer architecture settings -- lite_bottleneck
_C.MODEL.MF.LITE_BNECK =  CN()
_C.MODEL.MF.LITE_BNECK.IN_CHANNEL = 8
_C.MODEL.MF.LITE_BNECK.HIDDEN_CHANNEL = 24
_C.MODEL.MF.LITE_BNECK.OUT_CHANNEL = 12
_C.MODEL.MF.LITE_BNECK.KERNEL = 3
_C.MODEL.MF.LITE_BNECK.STRIED = 2
_C.MODEL.MF.LITE_BNECK.PADDING = 1

# mobileformer architecture settings -- block, defualt 26m
_C.MODEL.MF.BLOCK =  CN()
_C.MODEL.MF.BLOCK.IN_CHANNELS = [12, 12, 24, 24, 48, 48, 64, 96]
_C.MODEL.MF.BLOCK.HIDDEN_CHANNELS = [36, 72, 72, 144, 192, 288, 384, 576]
_C.MODEL.MF.BLOCK.OUT_CHANNELS = [12, 24, 24, 48, 48, 64, 96, 96]
_C.MODEL.MF.BLOCK.KERNELS = [3, 3, 3, 3, 3, 3, 3, 3]
_C.MODEL.MF.BLOCK.STRIEDS = [1, 2, 1, 2, 1, 1, 2, 1]
_C.MODEL.MF.BLOCK.PADDINGS = [1, 1, 1, 1, 1, 1, 1, 1]

# mobileformer architecture settings -- channel conv1x1
_C.MODEL.MF.CHANNEL_CONV =  CN()
_C.MODEL.MF.CHANNEL_CONV.IN_CHANNEL = 96
_C.MODEL.MF.CHANNEL_CONV.OUT_CHANNEL = 576

# mobileformer architecture settings -- conv1x1
_C.MODEL.MF.HEAD =  CN()
_C.MODEL.MF.HEAD.IN_CHANNEL = 96
_C.MODEL.MF.HEAD.HIDDEN_FEATURE = 576


# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 450
_C.TRAIN.WARMUP_EPOCHS = 30
_C.TRAIN.WEIGHT_DECAY = 0.08
_C.TRAIN.BASE_LR = 1e-2
_C.TRAIN.WARMUP_START_LR = 1e-7
_C.TRAIN.END_LR = 1e-3
_C.TRAIN.GRAD_CLIP = 2.0 # Clip gradient norm
_C.TRAIN.ACCUM_ITER = 1 # Gradient accumulation steps
_C.TRAIN.LINEAR_SCALED_LR = None

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# model ema
_C.TRAIN.MODEL_EMA = False
_C.TRAIN.MODEL_EMA_DECAY = 0.99996
_C.TRAIN.MODEL_EMA_FORCE_CPU = True

# data augmentation (optional, check datasets.py)
_C.TRAIN.SMOOTHING = 0.1
_C.TRAIN.COLOR_JITTER = 0.4  # if both auto augment and rand augment are False, use color jitter
_C.TRAIN.AUTO_AUGMENT = False  # rand augment is used if both rand and auto augment are set True
_C.TRAIN.RAND_AUGMENT = True
_C.TRAIN.RAND_AUGMENT_LAYERS = 2
_C.TRAIN.RAND_AUGMENT_MAGNITUDE = 9  # scale from 0 to 9
# mixup params (optional, check datasets.py)
_C.TRAIN.MIXUP_ALPHA = 0.8
_C.TRAIN.MIXUP_PROB = 1.0
_C.TRAIN.MIXUP_SWITCH_PROB = 0.5
_C.TRAIN.MIXUP_MODE = 'batch'
_C.TRAIN.CUTMIX_ALPHA = 1.0
_C.TRAIN.CUTMIX_MINMAX = None
# random erase params (optional, check datasets.py)
_C.TRAIN.RANDOM_ERASE_PROB = 0.25
_C.TRAIN.RANDOM_ERASE_MODE = 'pixel'
_C.TRAIN.RANDOM_ERASE_COUNT = 1
_C.TRAIN.RANDOM_ERASE_SPLIT = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = False
_C.SAVE = "./output"
_C.TAG = 'default'
_C.SAVE_FREQ = 1 # Frequency to save checkpoint
_C.REPORT_FREQ  = 20 # Frequency to logging info
_C.VALIDATE_FREQ = 10 # freq to do validation
_C.SEED = 0 # Fixed random seed
_C.EVAL = False
_C.THROUGHPUT_MODE = False


def _update_config_from_file(config, cfg_file):
    """Load cfg file (.yaml) and update config object

    Args:
        config: config object
        cfg_file: config file (.yaml)
    Return:
        None
    """
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    """Update config by ArgumentParser
    Configs that are often used can be updated from arguments
    Args:
        args: ArgumentParser contains options
    Return:
        config: updated config
    """
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
        config.DATA.BATCH_SIZE_EVAL = args.batch_size
    if args.batch_size_eval:
        config.DATA.BATCH_SIZE_EVAL = args.batch_size_eval
    if args.image_size:
        config.DATA.IMAGE_SIZE = args.image_size
    if args.accum_iter:
        config.TRAIN.ACCUM_ITER = args.accum_iter
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.eval:
        config.EVAL = True
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.last_epoch:
        config.TRAIN.LAST_EPOCH = args.last_epoch
    if args.amp:  # only for training
        config.AMP = not config.EVAL
    # config.freeze()
    return config


def get_config(cfg_file=None):
    """Return a clone of config and optionally overwrite it from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
