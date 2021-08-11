import os
from yacs.config import CfgNode as CN
import yaml

_C = CN()
_C.BASE = ['']

# data settings
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 4  #train batch_size for single GPU
_C.DATA.BATCH_SIZE_VAL = 1 # val batch_size for single GPU
_C.DATA.DATASET = 'PascalContext' # dataset name
_C.DATA.DATA_PATH = '/home/ssd3/wutianyi/datasets/pascal_context'
_C.DATA.CROP_SIZE = (480,480) # input_size (training)
_C.DATA.NUM_CLASSES = 60  # 19 for cityscapes, 60 for Pascal-Context
_C.DATA.NUM_WORKERS = 0 # number of data loading threads (curren paddle must set to 0)

# model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'SETR_MLA'
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.TYPE = 'ViT_MLA'
_C.MODEL.ENCODER.OUT_INDICES = [5,11,17,23]

_C.MODEL.DECODER_TYPE = 'ViT_MLAHead'
_C.MODEL.RESUME = None
_C.MODEL.PRETRAINED = None
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROPOUT = 0.0 # 0.0
_C.MODEL.ATTENTION_DROPOUT = 0.0
_C.MODEL.DROP_PATH = 0.1  # for SwinTransformer

# Transformer backbone settings 
_C.MODEL.TRANS = CN()
_C.MODEL.TRANS.HYBRID = False       #TODO: implement
_C.MODEL.TRANS.PATCH_GRID = None    #TODO: implement
_C.MODEL.TRANS.PATCH_SIZE = 16
_C.MODEL.TRANS.HIDDEN_SIZE = 768  # 768(Base), 1024(Large), 1280(Huge)
_C.MODEL.TRANS.MLP_RATIO = 4
_C.MODEL.TRANS.NUM_HEADS = None      # 12(Base), 16(Large), 16(Huge)
_C.MODEL.TRANS.NUM_LAYERS = 12     # 12(Base), 24(Large), 32(Huge)
_C.MODEL.TRANS.QKV_BIAS = True

## special settings for SwinTransformer
_C.MODEL.TRANS.WINDOW_SIZE = 7
_C.MODEL.TRANS.IN_CHANNELS = 3
_C.MODEL.TRANS.EMBED_DIM = 96  # same as HIDDEN_SIZE
_C.MODEL.TRANS.STAGE_DEPTHS = [2, 2, 6, 2]
_C.MODEL.TRANS.NUM_HEADS = None     # [3, 6, 12, 24]
_C.MODEL.TRANS.QK_SCALE = None
_C.MODEL.TRANS.APE = False   # absolute postional embedding
_C.MODEL.TRANS.PATCH_NORM = True   
#_C.MODEL.TRANS.DROP_PATH_RATE = None   
_C.MODEL.TRANS.KEEP_CLS_TOKEN = False


# MLA Decoder setting
_C.MODEL.MLA = CN()
#_C.MODEL.MLA.MLA_INDEX = [2, 5, 8, 11]   # Base: [2, 5, 8, 11]; Large: [5, 11, 17, 23] 
_C.MODEL.MLA.MLA_CHANNELS = 256
_C.MODEL.MLA.MLAHEAD_CHANNELS=128
_C.MODEL.MLA.AUXIHEAD = False
_C.MODEL.MLA.MLAHEAD_ALIGN_CORNERS = False


# PUP and Naive Decoder setting
_C.MODEL.PUP = CN()
_C.MODEL.PUP.INPUT_CHANNEL = 1024
_C.MODEL.PUP.NUM_CONV = 4
_C.MODEL.PUP.NUM_UPSAMPLE_LAYER = 4
_C.MODEL.PUP.CONV3x3_CONV1x1 = True
_C.MODEL.PUP.ALIGN_CORNERS = False

# Auxi PUP and Naive Decoder setting
_C.MODEL.AUXPUP = CN()
_C.MODEL.AUXPUP.INPUT_CHANNEL = 1024
_C.MODEL.AUXPUP.NUM_CONV = 2
_C.MODEL.AUXPUP.NUM_UPSAMPLE_LAYER = 2
_C.MODEL.AUXPUP.CONV3x3_CONV1x1 = True
_C.MODEL.AUXPUP.ALIGN_CORNERS = False

# UperHead Decoder setting
_C.MODEL.UPERHEAD = CN()
_C.MODEL.UPERHEAD.IN_CHANNELS = [96, 192, 384, 768]
_C.MODEL.UPERHEAD.CHANNELS = 512
_C.MODEL.UPERHEAD.IN_INDEX = [0, 1, 2, 3]
_C.MODEL.UPERHEAD.POOL_SCALES = [1, 2, 3, 6]
_C.MODEL.UPERHEAD.DROP_RATIO = 0.1
_C.MODEL.UPERHEAD.ALIGN_CORNERS = False

# Auxilary Segmentation Head setting
_C.MODEL.AUX = CN()
_C.MODEL.AUX.AUXIHEAD = True
_C.MODEL.AUX.AUXHEAD_ALIGN_CORNERS = False

# Auxilary FCN Head
_C.MODEL.AUXFCN = CN()
_C.MODEL.AUXFCN.IN_CHANNELS = 384
_C.MODEL.AUXFCN.UP_RATIO = 16

#DPT Head settings
_C.MODEL.DPT = CN()
_C.MODEL.DPT.HIDDEN_FEATURES = [256, 512, 1024, 1024]
_C.MODEL.DPT.FEATURES = 256
_C.MODEL.DPT.READOUT_PROCESS = "project"

#Segmenter Head Settings
_C.MODEL.SEGMENTER = CN()
_C.MODEL.SEGMENTER.NUM_LAYERS = 2

# training settings
_C.TRAIN = CN()
_C.TRAIN.USE_GPU = True
_C.TRAIN.LAST_EPOCH = 0
_C.TRAIN.BASE_LR = 0.001 #0.003 for pretrain # 0.03 for finetune
_C.TRAIN.END_LR = 1e-4
_C.TRAIN.DECODER_LR_COEF = 1.0
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.ITERS = 80000
_C.TRAIN.WEIGHT_DECAY = 0.0 # 0.0 for finetune
_C.TRAIN.POWER=0.9
_C.TRAIN.DECAY_STEPS= 80000

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'PolynomialDecay'
_C.TRAIN.LR_SCHEDULER.MILESTONES = "30, 60, 90" # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30 # only used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1 # only used in StepLRScheduler

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'SGD'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)  # for adamW
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# val settings
_C.VAL = CN()
_C.VAL.USE_GPU = True
_C.VAL.MULTI_SCALES_VAL = False
_C.VAL.SCALE_RATIOS= [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
_C.VAL.IMAGE_BASE_SIZE = None # 520 for pascal context
_C.VAL.KEEP_ORI_SIZE = False
_C.VAL.RESCALE_FROM_ORI = False
_C.VAL.CROP_SIZE = [480,480]
_C.VAL.STRIDE_SIZE = [320,320]
_C.VAL.MEAN = [123.675, 116.28, 103.53]
_C.VAL.STD = [58.395, 57.12, 57.375]

# misc
_C.SAVE_DIR = "./output"
_C.KEEP_CHECKPOINT_MAX = 3
_C.TAG = "default"
_C.SAVE_FREQ_CHECKPOINT = 1000 # freq to save chpt
_C.LOGGING_INFO_FREQ = 50 # freq to logging info
_C.VALIDATE_FREQ = 2000 # freq to do validation
_C.SEED = 0
_C.EVAL = False # run evaluation only
_C.LOCAL_RANK = 0

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
    if args.cfg:
        _update_config_from_file(config, args.cfg)
    config.defrost()

    config.freeze()
    return config

def get_config():
    config = _C.clone()
    return config
