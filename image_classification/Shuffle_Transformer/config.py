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

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 8 #1024 batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 8 #1024 batch_size for single GPU
_C.DATA.DATA_PATH = '/dataset/imagenet/' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size
_C.DATA.CROP_PCT = 0.9 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 4 # number of data loading threads
_C.DATA.IMAGENET_MEAN = [0.485, 0.456, 0.406] # [0.5, 0.5, 0.5]
_C.DATA.IMAGENET_STD = [0.229, 0.224, 0.225] # [0.5, 0.5, 0.5]

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'ShuffleTransformer'
_C.MODEL.NAME = 'ShuffleTransformer'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.0
_C.MODEL.ATTENTION_DROPOUT = 0.0
_C.MODEL.DROP_PATH = 0.1

# transformer settings
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.PATCH_SIZE = 4 # image_size = patch_size x window_size x num_windows
_C.MODEL.TRANS.WINDOW_SIZE = 7
_C.MODEL.TRANS.IN_CHANNELS = 3
_C.MODEL.TRANS.EMBED_DIM = 96 # same as HIDDEN_SIZE in ViT
_C.MODEL.TRANS.DEPTHS = [2, 2, 6, 2]
_C.MODEL.TRANS.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.TRANS.MLP_RATIO = 4.
_C.MODEL.TRANS.QKV_BIAS = True
_C.MODEL.TRANS.QK_SCALE = None
_C.MODEL.TRANS.APE = False # absolute positional embeddings
_C.MODEL.TRANS.PATCH_NORM = True

# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WARMUP_START_LR = 5e-7
_C.TRAIN.END_LR = 5e-6
_C.TRAIN.GRAD_CLIP = 5.0
_C.TRAIN.ACCUM_ITER = 1
_C.TRAIN.LINEAR_SCALED_LR = None

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'warmupcosine'
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)  # for adamW
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# train augmentation
_C.TRAIN.MIXUP_ALPHA = 0.8
_C.TRAIN.CUTMIX_ALPHA = 1.0
_C.TRAIN.CUTMIX_MINMAX = None
_C.TRAIN.MIXUP_PROB = 1.0
_C.TRAIN.MIXUP_SWITCH_PROB = 0.5
_C.TRAIN.MIXUP_MODE = 'batch'

_C.TRAIN.SMOOTHING = 0.1
_C.TRAIN.COLOR_JITTER = 0.4
_C.TRAIN.AUTO_AUGMENT = False #'rand-m9-mstd0.5-inc1'
_C.TRAIN.RAND_AUGMENT = False

_C.TRAIN.RANDOM_ERASE_PROB = 0.25
_C.TRAIN.RANDOM_ERASE_MODE = 'pixel'
_C.TRAIN.RANDOM_ERASE_COUNT = 1
_C.TRAIN.RANDOM_ERASE_SPLIT = False

# misc
_C.SAVE = "./output"
_C.TAG = "default"
_C.SAVE_FREQ = 1 # freq to save chpt
_C.REPORT_FREQ = 50 # freq to logging info
_C.VALIDATE_FREQ = 10 # freq to do validation
_C.SEED = 42
_C.EVAL = False # run evaluation only
_C.AMP = False
_C.LOCAL_RANK = 0
_C.NGPUS = -1


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('merging config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    """Update config by ArgumentParser
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
    if args.eval:
        config.EVAL = True
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
        if config.EVAL:
            config.DATA.BATCH_SIZE_EVAL = args.batch_size
    if args.image_size:
        config.DATA.IMAGE_SIZE = args.image_size
    if args.num_classes:
        config.MODEL.NUM_CLASSES = args.num_classes
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.output is not None:
        config.SAVE = args.output
    if args.ngpus:
        config.NGPUS = args.ngpus
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.last_epoch:
        config.TRAIN.LAST_EPOCH = args.last_epoch
    if args.amp: # only during training
        if config.EVAL is True:
            config.AMP = False
        else:
            config.AMP = True

    #config.freeze()
    return config


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
