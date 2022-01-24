import numpy as np
import paddle
from yacs.config import CfgNode as CN

config = CN()
config.ATSS = CN()

config.ATSS.ASPECT_RATIOS = [1.0]
config.ATSS.ANCHOR_SIZE = [64, 128, 256, 512, 1024]
config.ATSS.STRIDES = [8, 16, 32, 64, 128]
config.ATSS.OFFSET = 0
config.ATSS.NUM_CLASSES = 80
config.ATSS.FOCAL_LOSS_ALPHA = 0.25
config.ATSS.FOCAL_LOSS_GAMMA = 2
config.ATSS.REG_LOSS_WEIGHT = 2.0
config.ATSS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
config.ATSS.NUM_CONVS = 4
config.ATSS.INPUT_CHANNELS = 256
config.ATSS.PRIOR_PROB = 0.01
config.ATSS.TOPK = 9
config.ATSS.SCORE_THRESH = 0.05
config.ATSS.KEEP_TOPK = 100
config.ATSS.NMS_TOPK = 1000
config.ATSS.NMS_THRESH = 0.6

