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
import yaml
from yacs.config import CfgNode as CN

_C = CN()
# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings -- ok
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128 #1024 batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 128 #1024 batch_size for single GPU
_C.DATA.DATA_PATH = '/home/aistudio/ILSVRC2012_val' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size
_C.DATA.CROP_PCT = 0.875 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 4 # number of data loading threads
_C.DATA.IMAGENET_MEAN = [0.485, 0.456, 0.406] # [0.5, 0.5, 0.5]
_C.DATA.IMAGENET_STD = [0.229, 0.224, 0.225] # [0.5, 0.5, 0.5]


# -----------------------------------------------------------------------------
# Model settings -- maybe ok
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'focal'
_C.MODEL.NAME = 'focal_tiny_patch4_window7_224'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.0
_C.MODEL.ATTENTION_DROPOUT = 0.0
_C.MODEL.DROP_PATH = 0.1


# Focal Transformer parameters
# These hyperparams are the same to Swin Transformer, but we do not use shift by default
_C.MODEL.FOCAL = CN()
_C.MODEL.FOCAL.PATCH_SIZE = 4
_C.MODEL.FOCAL.IN_CHANS = 3
_C.MODEL.FOCAL.EMBED_DIM = 96
_C.MODEL.FOCAL.DEPTHS = [2, 2, 6, 2]
_C.MODEL.FOCAL.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.FOCAL.WINDOW_SIZE = 7
_C.MODEL.FOCAL.MLP_RATIO = 4.
_C.MODEL.FOCAL.QKV_BIAS = True
_C.MODEL.FOCAL.QK_SCALE = False
_C.MODEL.FOCAL.APE = False
_C.MODEL.FOCAL.PATCH_NORM = True
_C.MODEL.FOCAL.USE_SHIFT = False


# Below are specifical for Focal Transformers
_C.MODEL.FOCAL.FOCAL_POOL = "none"
_C.MODEL.FOCAL.FOCAL_STAGES = [0, 1, 2, 3]
_C.MODEL.FOCAL.FOCAL_LEVELS = [1, 1, 1, 1]
_C.MODEL.FOCAL.FOCAL_WINDOWS = [7, 5, 3, 1]
_C.MODEL.FOCAL.EXPAND_STAGES = [0, 1, 2, 3]
_C.MODEL.FOCAL.EXPAND_SIZES = [3, 3, 3, 3]
_C.MODEL.FOCAL.EXPAND_LAYER = "all"
_C.MODEL.FOCAL.USE_CONV_EMBED = False
_C.MODEL.FOCAL.USE_LAYERSCALE = False
_C.MODEL.FOCAL.USE_PRE_NORM = False


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_START_LR = 5e-7
_C.TRAIN.END_LR = 5e-6
_C.TRAIN.GRAD_CLIP = 5.0 # Clip gradient norm
_C.TRAIN.ACCUM_ITER = 1 # Gradient accumulation steps


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'warmupcosine' # origin is cosine
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler


# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.TRAIN.MIXUP_ALPHA = 0.8
_C.TRAIN.CUTMIX_ALPHA = 1.0
_C.TRAIN.CUTMIX_MINMAX = None
_C.TRAIN.MIXUP_PROB = 1.0
_C.TRAIN.MIXUP_SWITCH_PROB = 0.5
_C.TRAIN.MIXUP_MODE = 'batch'

_C.TRAIN.SMOOTHING = 0.1
_C.TRAIN.COLOR_JITTER = 0.4
_C.TRAIN.AUTO_AUGMENT = True #'rand-m9-mstd0.5-inc1'

_C.TRAIN.RANDOM_ERASE_PROB = 0.25
_C.TRAIN.RANDOM_ERASE_MODE = 'pixel' # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.TRAIN.RANDOM_ERASE_COUNT = 1
_C.TRAIN.RANDOM_ERASE_SPLIT = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True   # 预测时，是否使用裁剪

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = False
_C.SAVE = "./output"
_C.TAG = 'default'
_C.SAVE_FREQ = 1 # Frequency to save checkpoint
_C.REPORT_FREQ  = 100 # Frequency to logging info
_C.VALIDATE_FREQ = 10 # freq to do validation
_C.SEED = 0 # Fixed random seed
_C.EVAL = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0
_C.NGPUS = -1


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    """Update config by ArgumentParser
    Args:
        args: ArgumentParser contains options
    Return:
        config: updated config
    """
    _update_config_from_file(config, args.cfg)

    config.defrost()
    # merge from specific arguments
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.image_size:
        config.DATA.IMAGE_SIZE = args.image_size
    if args.num_classes:
        config.MODEL.NUM_CLASSES = args.num_classes
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.ngpus:
        config.NGPUS = args.ngpus
    if args.eval:
        config.EVAL = True
        config.DATA.BATCH_SIZE_EVAL = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.last_epoch:
        config.TRAIN.LAST_EPOCH = args.last_epoch
    if args.output is not None:
        config.SAVE = args.output 
    if args.save_freq:
        config.SAVE_FREQ = args.save_freq
    if args.log_freq:
        config.REPORT_FREQ = args.log_freq
    if args.validate_freq:
        config.VALIDATE_FREQ = args.validate_freq 
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.accum_iter: 
        config.TRAIN.ACCUM_ITER = args.accum_iter
    if args.amp: # only during training
        if config.EVAL is True:
            config.AMP = False
        else:
            config.AMP = True
    
    # output folder
    config.SAVE = os.path.join(config.SAVE, config.MODEL.NAME, config.TAG)
    # config.freeze()

    return config


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config