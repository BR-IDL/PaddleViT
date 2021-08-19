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
config.RETINANET = CN()

config.RETINANET.NUM_CONVS = 4
config.RETINANET.INPUT_CHANNELS = 256
config.RETINANET.NORM = ""
config.RETINANET.PRIOR_PROB = 0.01
config.RETINANET.NUM_CLASSES = 80
config.RETINANET.FOCAL_LOSS_ALPHA = 0.25
config.RETINANET.FOCAL_LOSS_GAMMA = 2
config.RETINANET.SMOOTHL1_LOSS_DELTA = 0
config.RETINANET.POSITIVE_THRESH = 0.5
config.RETINANET.NEGATIVE_THRESH = 0.4
config.RETINANET.ALLOW_LOW_QUALITY = True
config.RETINANET.WEIGHTS = [1.0, 1.0, 1.0, 1.0]
config.RETINANET.SCORE_THRESH = 0.05
config.RETINANET.KEEP_TOPK = 100
config.RETINANET.NMS_TOPK = 1000
config.RETINANET.NMS_THRESH = 0.5
config.RETINANET.ANCHOR_SIZE = [[x, x * 2**(1.0/3), x * 2**(2.0/3)] for x in [32, 64, 128, 256, 512 ]]
config.RETINANET.ASPECT_RATIOS = [0.5, 1.0, 2.0]
config.RETINANET.STRIDES = [8.0, 16.0, 32.0, 64.0, 128.0]
config.RETINANET.OFFSET = 0