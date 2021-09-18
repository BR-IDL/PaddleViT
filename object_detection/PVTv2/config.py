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
_C.DATA.BATCH_SIZE = 8 #1024 batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 1 #1024 batch_size for single GPU
_C.DATA.WEIGHT_PATH = './weights/pvtv2_b0_maskrcnn.pdparams' #"./weights/mask_rcnn_swin_small_patch4_window7.pdparams"
_C.DATA.VAL_DATA_PATH = "/dataset/coco/" # path to dataset
_C.DATA.DATASET = 'coco' # dataset name
_C.DATA.IMAGE_SIZE = 640 # input image size
_C.DATA.CROP_PCT = 0.9 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 1 # number of data loading threads

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'PVTv2_Det'
_C.MODEL.NAME = 'pvtv2_maskrcnn_b0'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.0
_C.MODEL.ATTENTION_DROPOUT = 0.0
_C.MODEL.DROP_PATH = 0.0 # TODO: droppath may raise cuda error on paddle.rand method

# transformer settings
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.PRETRAIN_IMAGE_SIZE = 224
_C.MODEL.TRANS.PATCH_SIZE = 4 # image_size = patch_size x window_size x num_windows
_C.MODEL.TRANS.IN_CHANNELS = 3
_C.MODEL.TRANS.EMBED_DIMS = [32, 64, 160, 256] 
_C.MODEL.TRANS.STAGE_DEPTHS = [2, 2, 2, 2]   
_C.MODEL.TRANS.NUM_HEADS = [1, 2, 5, 8]
_C.MODEL.TRANS.MLP_RATIO = [8, 8, 4, 4]
_C.MODEL.TRANS.SR_RATIO = [8, 4, 2, 1]
_C.MODEL.TRANS.QKV_BIAS = True
_C.MODEL.TRANS.QK_SCALE = None
_C.MODEL.TRANS.LINEAR = False
_C.MODEL.TRANS.OUT_INDICES = (1, 2, 3) # for maskrcnn (0, 1, 2, 3) [32, 64, 160, 256]
_C.MODEL.TRANS.FROZEN_STAGES = -1


# fpn settings
_C.FPN = CN()
_C.FPN.OUT_CHANNELS = 256
_C.FPN.IN_CHANNELS = [64, 160, 256]  # for maskrcnn [32, 64, 160, 256] [256, 512, 1024, 2048]
_C.FPN.USE_C5 = True
_C.FPN.STRIDES = [8, 16, 32] # for maskrcnn [4, 8, 16, 32]

# maskrcnn_head settings
_C.RPN = CN()
_C.ROI = CN()
_C.ROI.BOX_HEAD = CN()

_C.RPN.ANCHOR_SIZE = [[32], [64], [128], [256], [512]]
_C.RPN.ASPECT_RATIOS = [0.5, 1.0, 2.0]
_C.RPN.STRIDES = [4, 8, 16, 32, 64]
_C.RPN.OFFSET = 0.0
_C.RPN.PRE_NMS_TOP_N_TRAIN = 2000
_C.RPN.POST_NMS_TOP_N_TRAIN = 1000
_C.RPN.PRE_NMS_TOP_N_TEST = 1000
_C.RPN.POST_NMS_TOP_N_TEST = 1000
_C.RPN.NMS_THRESH = 0.7
_C.RPN.MIN_SIZE = 0.0
_C.RPN.TOPK_AFTER_COLLECT = True
_C.RPN.POSITIVE_THRESH = 0.7
_C.RPN.NEGATIVE_THRESH = 0.3
_C.RPN.BATCH_SIZE_PER_IMG = 256
_C.RPN.POSITIVE_FRACTION = 0.5
_C.RPN.LOW_QUALITY_MATCHES = True

_C.ROI.SCORE_THRESH_INFER = 0.05
_C.ROI.NMS_THRESH_INFER = 0.5
_C.ROI.NMS_KEEP_TOPK_INFER = 100
_C.ROI.NUM_ClASSES = 80
_C.ROI.POSITIVE_THRESH = 0.5
_C.ROI.NEGATIVE_THRESH = 0.5
_C.ROI.BATCH_SIZE_PER_IMG = 512
_C.ROI.POSITIVE_FRACTION = 0.25
_C.ROI.LOW_QUALITY_MATCHES = False
_C.ROI.BOX_HEAD.REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
_C.ROI.BOX_HEAD.NUM_CONV = 0
_C.ROI.BOX_HEAD.CONV_DIM = 256
_C.ROI.BOX_HEAD.NUM_FC = 2
_C.ROI.BOX_HEAD.FC_DIM = 1024
_C.ROI.SCALES = [1./4., 1./8., 1./16., 1./32, 1./64.]
_C.ROI.ALIGN_OUTPUT_SIZE = 7
_C.ROI.SAMPLING_RATIO = 0
_C.ROI.CANONICAL_BOX_SIZE = 224
_C.ROI.CANONICAL_LEVEL = 4
_C.ROI.MIN_LEVEL = 0
_C.ROI.MAX_LEVEL = 3
_C.ROI.ALIGNED = True
_C.ROI.PAT_GT_AS_PRO = True # when eval, set to False

# retinanet_head setting
_C.RETINANET = CN()
_C.RETINANET.NUM_CONVS = 4
_C.RETINANET.INPUT_CHANNELS = 256
_C.RETINANET.NORM = ""
_C.RETINANET.PRIOR_PROB = 0.01
_C.RETINANET.NUM_CLASSES = 80
_C.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.RETINANET.FOCAL_LOSS_GAMMA = 2
_C.RETINANET.SMOOTHL1_LOSS_DELTA = 0
_C.RETINANET.POSITIVE_THRESH = 0.5
_C.RETINANET.NEGATIVE_THRESH = 0.4
_C.RETINANET.ALLOW_LOW_QUALITY = True
_C.RETINANET.WEIGHTS = [1.0, 1.0, 1.0, 1.0]
_C.RETINANET.SCORE_THRESH = 0.05
_C.RETINANET.KEEP_TOPK = 100
_C.RETINANET.NMS_TOPK = 1000
_C.RETINANET.NMS_THRESH = 0.5
_C.RETINANET.ANCHOR_SIZE = [[x, x * 2**(1.0/3), x * 2**(2.0/3)] for x in [32, 64, 128, 256, 512]]
_C.RETINANET.ASPECT_RATIOS = [0.5, 1.0, 2.0]
_C.RETINANET.STRIDES = [8.0, 16.0, 32.0, 64.0, 128.0]
_C.RETINANET.OFFSET = 0

# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WARMUP_START_LR = 0.0
_C.TRAIN.END_LR = 0.0
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.ACCUM_ITER = 2

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'warmupcosine'
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'SGD'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# augmentation
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4 # color jitter factor
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.RE_PROB = 0.25 # random earse prob
_C.AUG.RE_MODE = 'pixel' # random earse mode
_C.AUG.RE_COUNT = 1 # random earse count
_C.AUG.MIXUP = 0.8 # mixup alpha, enabled if >0
_C.AUG.CUTMIX = 1.0 # cutmix alpha, enabled if >0
_C.AUG.CUTMIX_MINMAX = None # cutmix min/max ratio, overrides alpha
_C.AUG.MIXUP_PROB = 1.0 # prob of mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5 # prob of switching cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_MODE = 'batch' #how to apply mixup/curmix params, per 'batch', 'pair', or 'elem'

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
        config.MODEL.LAST_EPOCH = args.last_epoch

    #config.freeze()
    return config


def get_config(cfg_file=None):
    """Return a clone config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
