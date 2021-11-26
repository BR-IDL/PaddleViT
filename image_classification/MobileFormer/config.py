#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
_C.DATA.BATCH_SIZE_EVAL = 128 # val batch_size for single GPU
_C.DATA.DATA_PATH = 'ILSVRC2012_val/' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size: 224 for pretrain, 384 for finetune
_C.DATA.CROP_PCT = 0.875 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 2 # number of data loading threads 

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
_C.TRAIN.BASE_LR = 8e-4
_C.TRAIN.WARMUP_START_LR = 1e-7
_C.TRAIN.END_LR = 1e-5
_C.TRAIN.GRAD_CLIP = 2.0 # Clip gradient norm
_C.TRAIN.ACCUM_ITER = 1 # Gradient accumulation steps

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
    if args.model_type:
        config.MODEL.MF.TYPE = args.model_type
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
    if args.output:
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

    # output folder
    config.SAVE = os.path.join(config.SAVE, config.MODEL.NAME, config.TAG)

    return config


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config