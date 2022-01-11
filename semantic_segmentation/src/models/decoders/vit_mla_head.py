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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class VIT_MLAHead(nn.Layer):
    """VIT_MLAHead

    VIT_MLAHead is the decoder of SETR-MLA, Ref https://arxiv.org/pdf/2012.15840.pdf.
    """

    def __init__(self, mla_channels=256, mlahead_channels=128, 
                 num_classes=60, align_corners=False):
        super(VIT_MLAHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        sync_norm_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        self.head2 = nn.Sequential(
            nn.Conv2D(
                mla_channels,
                mlahead_channels,
                3, 
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU(),
            nn.Conv2D(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU())
        self.head3 = nn.Sequential(
            nn.Conv2D(
                mla_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU(),
            nn.Conv2D(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU())
        self.head4 = nn.Sequential(
            nn.Conv2D(
                mla_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU(),
            nn.Conv2D(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU())
        self.head5 = nn.Sequential(
            nn.Conv2D(
                mla_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU(),
            nn.Conv2D(
                mlahead_channels,
                mlahead_channels,
                3,
                padding=1,
                bias_attr=False),
            nn.SyncBatchNorm(
                mlahead_channels,
                weight_attr=self.get_sync_norm_weight_attr(),
                bias_attr=sync_norm_bias_attr),
            nn.ReLU())
        self.cls = nn.Conv2D(4*mlahead_channels, self.num_classes, 3, padding=1)

    def get_sync_norm_weight_attr(self):
        return paddle.ParamAttr(
            initializer=nn.initializer.Uniform(low=0.0, high=1.0, name=None)) 

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        up4x_resolution = [4*item for item in mla_p2.shape[2:]]
        up16x_resolution = [16*item for item in mla_p2.shape[2:]]
        # head2: 2 Conv layers + 4x_upsmaple
        h2_out = self.head2(mla_p2)
        h2_out_x4 = F.interpolate(h2_out, up4x_resolution, 
                                  mode='bilinear', align_corners=True)
        h3_out = self.head3(mla_p3)
        h3_out_x4 = F.interpolate(h3_out, up4x_resolution, 
                                  mode='bilinear', align_corners=True)
        h4_out = self.head4(mla_p4)
        h4_out_x4 = F.interpolate(h4_out, up4x_resolution, 
                                  mode='bilinear', align_corners=True)
        h5_out = self.head5(mla_p5)
        h5_out_x4 = F.interpolate(h5_out, up4x_resolution, 
                                  mode='bilinear', align_corners=True)
        # concatenating multi-head
        hout_concat = paddle.concat([h2_out_x4, h3_out_x4, 
                                     h4_out_x4, h5_out_x4], axis=1) 
        # pixel-level cls.
        pred = self.cls(hout_concat) # (B, num_classes, H/4, W/4)
        pred_full = F.interpolate(
            pred, up16x_resolution, mode='bilinear', 
            align_corners=self.align_corners)
        return pred_full
