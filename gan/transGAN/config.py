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

_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4 #256 # train batch_size for single GPU
_C.DATA.BATCH_SIZE_EVAL = 8 #64 # val batch_size for single GPU
_C.DATA.DATA_PATH = '' # path to dataset
_C.DATA.DATASET = 'cifar10' # dataset name
_C.DATA.IMAGE_SIZE = 32 # input image size: 224 for pretrain, 384 for finetune
_C.DATA.CROP_PCT = 0.875 # input image scale ratio, scale is applied before centercrop in eval mode
_C.DATA.NUM_WORKERS = 2 # number of data loading threads 

# model settings
_C.MODEL = CN()
_C.MODEL.TYPE = 'ViT'
_C.MODEL.NAME = 'ViT'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.1
_C.MODEL.ATTENTION_DROPOUT = 0.0

# transformer settings
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.HYBRID = False #TODO(zhuyu): implement
_C.MODEL.TRANS.PATCH_GRID = None #TODO(zhuyu): implement
_C.MODEL.TRANS.PATCH_SIZE = 32
_C.MODEL.TRANS.HIDDEN_SIZE = 768
_C.MODEL.TRANS.MLP_DIM = 3072
_C.MODEL.TRANS.NUM_HEADS = 12
_C.MODEL.TRANS.NUM_LAYERS = 12
_C.MODEL.TRANS.QKV_BIAS = True

# training settings
_C.TRAIN = CN()
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 3 #34 # ~ 10k steps for 4096 batch size
_C.TRAIN.WEIGHT_DECAY = 0.05 #0.3 # 0.0 for finetune
_C.TRAIN.BASE_LR = 0.001 #0.003 for pretrain # 0.03 for finetune
_C.TRAIN.WARMUP_START_LR = 1e-6 #0.0
_C.TRAIN.END_LR = 5e-4
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
_C.SAVE_FREQ = 10 # freq to save chpt
_C.REPORT_FREQ = 100 # freq to logging info
_C.VALIDATE_FREQ = 100 # freq to do validation
_C.SEED = 0
_C.EVAL = False # run evaluation only
_C.LOCAL_RANK = 0
_C.NGPUS = -1

_C.world_size = 1
_C.rank = 0
_C.loca_rank = -1
_C.dist_url = "tcp://localhost:14256"
_C.dist_backend = "nccl"
_C.seed = 12345
_C.gpu = None
_C.multiprocessing_distributed = True
_C.max_epoch = 200
_C.max_iter = 500000
_C.gen_batch_size = 128
_C.dis_batch_size = 64
_C.g_lr = 0.0001
_C.wd = 0.001
_C.d_lr = 0.0001
_C.ctrl_lr = 0.00035
_C.lr_decay = False
_C.beta1 = 0.0
_C.beta2 = 0.99
_C.num_workers = 16
_C.latent_dim = 256
_C.img_size = 32
_C.channels = 3
_C.n_critic = 4
_C.val_freq = 20
_C.print_freq = 50
_C.load_path = None
_C.exp_name = "cifar_train"
_C.d_spectral_norm = False
_C.g_spectral_norm = False
_C.dataset = "cifar10"
_C.data_path = "./data"
_C.init_type = "xavier_uniform"
_C.gf_dim = 1024
_C.df_dim = 384
_C.gen_model = "ViT_custom"
_C.dis_model = "ViT_custom_scale2"
_C.controller = "controller"
_C.eval_batch_size = 8
_C.num_eval_imgs = 20000
_C.bottom_width = 8
_C.random_seed = 12345
_C.shared_epoch = 15
_C.grow_step1 = 25
_C.grow_step2 = 55
_C.max_search_iter = 90
_C.ctrl_step = 30
_C.ctrl_sample_batch = 1
_C.hid_size = 100
_C.baseline_decay = 0.9
_C.rl_num_eval_img = 5000
_C.num_candidate = 10
_C.topk = 5
_C.entropy_coeff = 0.001
_C.dynamic_reset_threshold = 0.001
_C.dynamic_reset_window = 500
_C.arch = None
_C.optimizer = "adam"
_C.loss = "wgangp-eps"
_C.n_classes = 0
_C.phi = 1.0
_C.grow_steps = [0,0]
_C.D_downsample = "avg"
_C.fade_in = 0.0
_C.d_depth = 7
_C.g_depth = "5,4,2"
_C.g_norm = "ln"
_C.d_norm = "ln"
_C.g_act = "gelu"
_C.d_act = "gelu"
_C.patch_size = 2
_C.fid_stat = "None"
_C.diff_aug = ""
_C.accumulated_times = 1
_C.g_accumulated_times = 1
_C.num_landmarks = 64
_C.d_heads = 4
_C.dropout = 0.0
_C.ema = 0.9999
_C.ema_warmup = 0.1
_C.ema_kimg = 500
_C.latent_norm = False
_C.ministd = False
_C.g_mlp = 4
_C.d_mlp = 4
_C.g_window_size = 8
_C.d_window_size = 8
_C.show = False

# _C.world_size = 1
# _C.rank = 0
# _C.loca_rank = -1
# _C.dist_url = "tcp://localhost:14256"
# _C.dist_backend = "nccl"
# _C.seed = 12345
# _C.gpu = None
# _C.multiprocessing_distributed = True
# _C.max_epoch = 200
# _C.max_iter = 500000
# _C.gen_batch_size = 128
# _C.dis_batch_size = 64
# _C.g_lr = 0.0001
# _C.wd = 0.001
# _C.d_lr = 0.0001
# _C.ctrl_lr = 0.00035
# _C.lr_decay = False
# _C.beta1 = 0.0
# _C.beta2 = 0.99
# _C.num_workers = 16
# _C.latent_dim = 256
# _C.img_size = 32
# _C.channels = 3
# _C.n_critic = 4
# _C.val_freq = 20
# _C.print_freq = 50
# _C.load_path = None
# _C.exp_name = "cifar_train"
# _C.d_spectral_norm = False
# _C.g_spectral_norm = False
# _C.dataset = "cifar10"
# _C.data_path = "./data"
# _C.init_type = "xavier_uniform"
# _C.gf_dim = 1024
# _C.df_dim = 384
# _C.gen_model = "ViT_custom"
# _C.dis_model = "ViT_custom_scale2"
# _C.controller = "controller"
# _C.eval_batch_size = 8
# _C.num_eval_imgs = 20000
# _C.bottom_width = 8
# _C.random_seed = 12345
# _C.shared_epoch = 15
# _C.grow_step1 = 25
# _C.grow_step2 = 55
# _C.max_search_iter = 90
# _C.ctrl_step = 30
# _C.ctrl_sample_batch = 1
# _C.hid_size = 100
# _C.baseline_decay = 0.9
# _C.rl_num_eval_img = 5000
# _C.num_candidate = 10
# _C.topk = 5
# _C.entropy_coeff = 0.001
# _C.dynamic_reset_threshold = 0.001
# _C.dynamic_reset_window = 500
# _C.arch = None
# _C.optimizer = "adam"
# _C.loss = "wgangp-eps"
# _C.n_classes = 0
# _C.phi = 1.0
# _C.grow_steps = [0,0]
# _C.D_downsample = "avg"
# _C.fade_in = 0.0
# _C.d_depth = 7
# _C.g_depth = "5,4,2"
# _C.g_norm = "ln"
# _C.d_norm = "ln"
# _C.g_act = "gelu"
# _C.d_act = "gelu"
# _C.patch_size = 2
# _C.fid_stat = "None"
# _C.diff_aug = ""
# _C.accumulated_times = 1
# _C.g_accumulated_times = 1
# _C.num_landmarks = 64
# _C.d_heads = 4
# _C.dropout = 0.0
# _C.ema = 0.9999
# _C.ema_warmup = 0.1
# _C.ema_kimg = 500
# _C.latent_norm = False
# _C.ministd = False
# _C.g_mlp = 4
# _C.d_mlp = 4
# _C.g_window_size = 8
# _C.d_window_size = 8
# _C.show = False

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
        config.MODEL.LAST_EPOCH = args.last_epoch

    #config.freeze()
    return config


def get_config():
    """Return a clone of config"""
    config = _C.clone()
    return config