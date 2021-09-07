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

"""Swin Transformer Object Detection"""

import paddle
import paddle.nn as nn
from config import get_config
from swin_backbone import SwinTransformer
from det_necks.fpn import FPN, LastLevelMaxPool
from det_heads.maskrcnn_head.rpn_head import RPNHead
from det_heads.maskrcnn_head.roi_head import RoIHead

cfg = get_config()

class SwinTransformerDet(nn.Layer):
    def __init__(self, config):
        super(SwinTransformerDet, self).__init__()
        self.backbone = SwinTransformer(config)
        self.neck = FPN(
            in_channels=config.FPN.IN_CHANNELS,
            out_channel=config.FPN.OUT_CHANNELS,
            strides=config.FPN.STRIDES,
            use_c5=config.FPN.USE_C5,
            top_block=LastLevelMaxPool()     
        )
        self.rpnhead = RPNHead(config)
        self.roihead = RoIHead(config)

        self.config = config
    
    def forward(self, x, gt=None):
        feats = self.neck(self.backbone(x.tensors))
        rpn_out = self.rpnhead(feats, gt)

        if self.config.ROI.PAT_GT:
            proposals = []
            for proposal, gt_box in zip(rpn_out[0], gt["gt_boxes"]):
                proposals.append(paddle.concat([proposal, gt_box]))
        else:
            proposals = rpn_out[0]


        final_out = self.roihead(feats, proposals, gt)

        if self.training:
            rpn_losses = rpn_out[2]
            # if training, final_out returns losses, now we combine the losses dicts
            final_out.update(rpn_losses)

        return final_out


def build_swin_det(config):
    model = SwinTransformerDet(config)
    return model
