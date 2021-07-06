import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import copy
import numpy as np
import math


class VIT_MLAHead(nn.Layer):
    """VIT_MLAHead 

    VIT_MLAHead is the decoder of SETR-MLA, Ref https://arxiv.org/pdf/2012.15840.pdf
    Reference:                                                                                                                                                
        Sixiao Zheng, et al. *"Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"*
    """

    def __init__(self, mla_channels=256, mlahead_channels=128, num_classes=60, align_corners=False):
        super(VIT_MLAHead, self).__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners
        sync_norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)) 

        self.head2 = nn.Sequential(# 3x3_Conv(256,128)
                                   nn.Conv2D(mla_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels, weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU(),
                                   # 3x3_Conv(128,128)
                                   nn.Conv2D(mlahead_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU())
        self.head3 = nn.Sequential(# 3x3_Conv(256,128)
                                   nn.Conv2D(mla_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU(),
                                   # 3x3_Conv(128,128)
                                   nn.Conv2D(mlahead_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU())
        self.head4 = nn.Sequential(# 3x3_Conv(256,128)
                                   nn.Conv2D(mla_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU(),
                                   # 3x3_Conv(128,128)
                                   nn.Conv2D(mlahead_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU())
        self.head5 = nn.Sequential(# 3x3_Conv(256,128)
                                   nn.Conv2D(mla_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU(),
                                   # 3x3_Conv(128,128)
                                   nn.Conv2D(mlahead_channels, mlahead_channels, 3, padding=1, bias_attr=False),
                                   nn.SyncBatchNorm(mlahead_channels,weight_attr=self.get_sync_norm_weight_attr(), bias_attr=sync_norm_bias_attr),
                                   nn.ReLU())

        self.cls = nn.Conv2D(4*mlahead_channels, self.num_classes, 3, padding=1)

    def get_sync_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=0.0, high=1.0, name=None)) 

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):

        up4x_resolution = [ 4*item for item in mla_p2.shape[2:]]
        up16x_resolution = [ 16*item for item in mla_p2.shape[2:]]
        # head2: 2 Conv layers + 4x_upsmaple
        h2_out = self.head2(mla_p2)
        h2_out_x4 = F.interpolate(h2_out, up4x_resolution, mode='bilinear', align_corners=True)
        # head3: ...
        h3_out = self.head3(mla_p3)
        h3_out_x4 = F.interpolate(h3_out, up4x_resolution, mode='bilinear', align_corners=True)
        # head4: ...
        h4_out = self.head4(mla_p4)
        h4_out_x4 = F.interpolate(h4_out, up4x_resolution, mode='bilinear', align_corners=True)
        # head5: ...
        h5_out = self.head5(mla_p5)
        h5_out_x4 = F.interpolate(h5_out, up4x_resolution, mode='bilinear', align_corners=True)
        # concatenating multi-head
        hout_concat = paddle.concat([h2_out_x4, h3_out_x4, h4_out_x4, h5_out_x4], axis=1) #(B,128*4,H/4,W/4)
        # pixel-level cls.
        pred = self.cls(hout_concat)  # (B, num_classes, H/4, W/4)
        pred_full = F.interpolate(pred, up16x_resolution, mode='bilinear', align_corners=self.align_corners)

        return pred_full


