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

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 256 #256 # train batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 8 #64 # val batch_size for single GPU
_C.DATA.DATA_PATH = '/dataset/imagenet/' # path to dataset
_C.DATA.DATASET = 'imagenet2012' # dataset name
_C.DATA.IMAGE_SIZE = 224 # input image size: 224 for pretrain, 384 for finetune
_C.DATA.CROP_PCT = 1.0 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 0 # number of data loading threads

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'RepMLP'
_C.MODEL.NAME = 'repmlpres50_light_224_train'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000


# transformer settings
_C.MODEL.MIXER = CN()
_C.MODEL.MIXER.NUM_BLOCKS=[3,4,6,3]
_C.MODEL.MIXER.BLOCK_TYPE='light'
_C.MODEL.MIXER.IMG_H=224
_C.MODEL.MIXER.IMG_W=224
_C.MODEL.MIXER.H=7
_C.MODEL.MIXER.W=7
_C.MODEL.MIXER.REPARAM_CONV_K=(1,3,5)
_C.MODEL.MIXER.FC1_FC2_REDUCTION=1
_C.MODEL.MIXER.FC3_GROUPS=4
_C.MODEL.MIXER.DEPLOY=False

# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 3 #34 # ~ 10k steps for 4096 batch size
_C.TRAIN.WEIGHT_DECAY = 0.01 #0.3 # 0.0 for finetune
_C.TRAIN.BASE_LR = 0.001 #0.003 for pretrain # 0.03 for finetune
_C.TRAIN.WARMUP_START_LR = 1e-6 #0.0
_C.TRAIN.END_LR = 1e-5
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.ACCUM_ITER = 2 #1

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

# misc
_C.SAVE = "./output"
_C.TAG = "default"
_C.SAVE_FREQ = 20 # freq to save chpt
_C.REPORT_FREQ = 50 # freq to logging info
_C.VALIDATE_FREQ = 20 # freq to do validation
_C.SEED = 0
_C.EVAL = False # run evaluation only
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
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.image_size:
        config.DATA.IMAGE_SIZE = args.image_size
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

    #config.freeze()
    return config


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
