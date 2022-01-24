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

import math

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from atss_loss import ATSSLoss
from ..det_utils.target_assign import AnchorGenerator
from ..retinanet_head.post_process import RetinaNetPostProcess

class Scale(nn.Layer):
    def __init__(self, init_value):
        super(Scale, self).__init__()
        self.scale = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_value)),
            dtype="float32"
        )

    def forward(self, inputs):
        out = inputs * self.scale
        return out


class ATSSHead(nn.Layer):
    def __init__(self, config):
        super(ATSSHead, self).__init__()
        in_channels= config.ATSS.INPUT_CHANNELS
        prior_prob = config.ATSS.PRIOR_PROB

        self.get_loss = ATSSLoss(
            aspect_ratios=config.ATSS.ASPECT_RATIOS,
            anchor_size=config.ATSS.ANCHOR_SIZE,
            focal_loss_alpha=config.ATSS.FOCAL_LOSS_ALPHA,
            focal_loss_gamma=config.ATSS.FOCAL_LOSS_GAMMA,
            topk=config.ATSS.TOPK,
            num_classes=config.ATSS.NUM_CLASSES,
            reg_loss_weight=config.ATSS.REG_LOSS_WEIGHT,
            bbox_reg_weights = config.ATSS.BBOX_REG_WEIGHTS
        )
        self.anchor_generator = AnchorGenerator(
            anchor_sizes=config.ATSS.ANCHOR_SIZE,
            aspect_ratios=config.ATSS.ASPECT_RATIOS,
            strides=config.ATSS.STRIDES,
            offset=config.ATSS.OFFSET
        )
        self.postprocess = RetinaNetPostProcess(
            num_classes=config.ATSS.NUM_CLASSES,
            score_threshold=config.ATSS.SCORE_THRESH,
            keep_top_k=config.ATSS.KEEP_TOPK,
            nms_top_k=config.ATSS.NMS_TOPK,
            nms_threshold=config.ATSS.NMS_THRESH,
            bbox_reg_weights=config.ATSS.BBOX_REG_WEIGHTS
        )
        self.num_classes = config.ATSS.NUM_CLASSES

        num_anchors = self.anchor_generator.num_anchors

        cls_convs = []
        reg_convs = []
        for i in range(config.ATSS.NUM_CONVS):
            cls_convs.append(
                nn.Conv2D(
                   in_channels=in_channels, 
                   out_channels=in_channels, 
                   kernel_size=3, 
                   stride=1, 
                   padding=1,
                   weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
                   bias_attr=paddle.ParamAttr(initializer=Constant(0.))
                )
            )

            cls_convs.append(nn.GroupNorm(32, in_channels))
            cls_convs.append(nn.ReLU())

            reg_convs.append(
                nn.Conv2D(
                   in_channels=in_channels, 
                   out_channels=in_channels, 
                   kernel_size=3, 
                   stride=1, 
                   padding=1,
                   weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
                   bias_attr=paddle.ParamAttr(initializer=Constant(0.))
                )
            )

            reg_convs.append(nn.GroupNorm(32, in_channels))
            reg_convs.append(nn.ReLU())

        self.add_sublayer("cls_convs", nn.Sequential(*cls_convs))
        self.add_sublayer("reg_convs", nn.Sequential(*reg_convs))

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_score = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(bias_value))
        )
        self.bbox_pred = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*4,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(0.))
        )
        self.centerness = nn.Conv2D(
            in_channels=in_channels, 
            out_channels=num_anchors*1,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(0.))
        )

        self.scales = nn.LayerList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, feats, inputs):
        anchors = self.anchor_generator(feats)

        cls_pred_list = []
        reg_pred_list = []
        centerness_pred_list = []

        for i, feat in enumerate(feats):
            cls_feat = self.cls_convs(feat)
            reg_feat = self.reg_convs(feat)

            cls_per_lvl = self.cls_score(cls_feat)
            reg_per_lvl = self.scales[i](self.bbox_pred(reg_feat))
            centerness_per_lvl = self.centerness(reg_feat)

            B, _, H, W = cls_per_lvl.shape
            cls_per_lvl = cls_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, self.num_classes])
            reg_per_lvl = reg_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, 4])
            centerness_per_lvl = centerness_per_lvl.transpose([0, 2, 3, 1]).reshape([B, -1, 1])
            cls_pred_list.append(cls_per_lvl)
            reg_pred_list.append(reg_per_lvl)
            centerness_pred_list.append(centerness_per_lvl)

        if self.training:
            loss_dict = self.get_loss(cls_pred_list, reg_pred_list, centerness_pred_list, anchors, inputs)
            return loss_dict
        
        else:
            pred_result, bbox_num = self.postprocess(
                pred_scores_list, 
                pred_boxes_list, 
                anchors,
                inputs,
            )

            return pred_result, bbox_num
