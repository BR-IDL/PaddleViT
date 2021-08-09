#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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


from yacs.config import CfgNode as CN

config = CN()
config.FPN = CN()
config.RPN = CN()
config.ROI = CN()
config.ROI.BOX_HEAD = CN()

config.FPN.OUT_CHANNELS = 256

config.RPN.ANCHOR_SIZE = [[32], [64], [128], [256], [512]]
config.RPN.ASPECT_RATIOS = [0.5, 1.0, 2.0]
config.RPN.STRIDES = [4, 8, 16, 32, 64]
config.RPN.OFFSET = 0.0
config.RPN.PRE_NMS_TOP_N_TRAIN = 2000
config.RPN.POST_NMS_TOP_N_TRAIN = 1000
config.RPN.PRE_NMS_TOP_N_TEST = 1000
config.RPN.POST_NMS_TOP_N_TEST = 1000
config.RPN.NMS_THRESH = 0.7
config.RPN.MIN_SIZE = 0.0
config.RPN.TOPK_AFTER_COLLECT = True
config.RPN.POSITIVE_THRESH = 0.7
config.RPN.NEGATIVE_THRESH = 0.3
config.RPN.BATCH_SIZE_PER_IMG = 256
config.RPN.POSITIVE_FRACTION = 0.5
config.RPN.LOW_QUALITY_MATCHES = True

config.ROI.SCORE_THRESH_INFER = 0.05
config.ROI.NMS_THRESH_INFER = 0.5
config.ROI.NMS_KEEP_TOPK_INFER =100
config.ROI.NUM_ClASSES = 80
config.ROI.POSITIVE_THRESH = 0.5
config.ROI.NEGATIVE_THRESH = 0.5
config.ROI.BATCH_SIZE_PER_IMG = 512
config.ROI.POSITIVE_FRACTION = 0.25
config.ROI.LOW_QUALITY_MATCHES = True
config.ROI.BOX_HEAD.REG_WEIGHTS = [10.0, 10.0, 5.0, 5.0]
config.ROI.BOX_HEAD.NUM_CONV = 0
config.ROI.BOX_HEAD.CONV_DIM = 256
config.ROI.BOX_HEAD.NUM_FC = 2
config.ROI.BOX_HEAD.FC_DIM = 1024
config.ROI.SCALES = [1./4., 1./8., 1./16., 1./32., 1./64.]
config.ROI.ALIGN_OUTPUT_SIZE = 7
config.ROI.SAMPLING_RATIO = 0
config.ROI.CANONICAL_BOX_SIZE = 224
config.ROI.CANONICAL_LEVEL = 4
config.ROI.MIN_LEVEL = 0
config.ROI.MAX_LEVEL = 3
config.ROI.ALIGNED = True
