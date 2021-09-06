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

from paddle.nn.initializer import Normal, Constant

from retinanet_loss import RetinaNetLoss
from post_process import RetinaNetPostProcess
from det_utils.generator_utils import AnchorGenerator

class RetinaNetHead(nn.Layer):
    '''
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    '''
    def __init__(self, config):
        '''
        Args:
            input_shape (List[ShapeSpec]): input shape.
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors.
            conv_dims (List[int]): dimensions for each convolution layer.
            norm (str or callable):
                    Normalization for conv layers except for the two output layers.
                    See :func:`detectron2.layers.get_norm` for supported types.
            loss_func (class): the class is used to compute loss.
            prior_prob (float): Prior weight for computing bias.
        '''
        super(RetinaNetHead, self).__init__()

        num_convs = config.RETINANET.NUM_CONVS
        input_channels = config.RETINANET.INPUT_CHANNELS
        norm = config.RETINANET.NORM
        prior_prob = config.RETINANET.PRIOR_PROB

        self.num_classes = config.RETINANET.NUM_CLASSES
        self.get_loss = RetinaNetLoss(
            focal_loss_alpha=config.RETINANET.FOCAL_LOSS_ALPHA,
            focal_loss_gamma=config.RETINANET.FOCAL_LOSS_GAMMA,
            smoothl1_loss_delta=config.RETINANET.SMOOTHL1_LOSS_DELTA,
            positive_thresh=config.RETINANET.POSITIVE_THRESH,
            negative_thresh=config.RETINANET.NEGATIVE_THRESH,
            allow_low_quality=config.RETINANET.ALLOW_LOW_QUALITY,
            num_classes=config.RETINANET.NUM_CLASSES,
            weights=config.RETINANET.WEIGHTS
        )
        self.postprocess = RetinaNetPostProcess(
            score_threshold=config.RETINANET.SCORE_THRESH,
            keep_top_k=config.RETINANET.KEEP_TOPK,
            nms_top_k=config.RETINANET.NMS_TOPK,
            nms_threshold=config.RETINANET.NMS_THRESH,
            bbox_reg_weights=config.RETINANET.WEIGHTS
        )
        self.anchor_generator = AnchorGenerator(anchor_sizes=config.RETINANET.ANCHOR_SIZE,
                                                aspect_ratios=config.RETINANET.ASPECT_RATIOS,
                                                strides=config.RETINANET.STRIDES,
                                                offset=config.RETINANET.OFFSET)

        num_anchors = self.anchor_generator.num_anchors
        conv_dims = [input_channels] * num_convs

        cls_net = []
        reg_net = []

        for in_channels, out_channels in zip(
            [input_channels] + list(conv_dims), conv_dims
        ):
            cls_net.append(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))
            )
            if norm == "bn":
                cls_net.append(nn.BatchNorm2D(out_channels))
            cls_net.append(nn.ReLU())

            reg_net.append(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))
            )
            if norm == "bn":
                reg_net.append(nn.BatchNorm2D(out_channels))
            reg_net.append(nn.ReLU())

        self.cls_net = nn.Sequential(*cls_net)
        self.reg_net = nn.Sequential(*reg_net)

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.cls_score = nn.Conv2D(
            conv_dims[-1], num_anchors * self.num_classes, kernel_size=3, stride=1, padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)),
            bias_attr=paddle.ParamAttr(initializer=Constant(bias_value))
        )
        self.bbox_pred = nn.Conv2D(
            conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1,
            weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01))
        )

    def forward(self, feats, inputs):
        '''
         Returns:
            loss_dict (dict) | pred_result(tensor), bbox_num(tensor): 
            loss_dict: contains cls_losses and reg_losses.
            pred_result: the shape is [M, 6], M is the number of final preds,
                Each row has 6 values: [label, score, xmin, ymin, xmax, ymax]
            bbox_num: the shape is [N], N is the num of batch_size, 
                bbox_num[i] means the i'th img have bbox_num[i] boxes.
        '''
        anchors = self.anchor_generator(feats)

        pred_scores = []
        pred_boxes = []

        for feat in feats:
            pred_scores.append(self.cls_score(self.cls_net(feat)))
            pred_boxes.append(self.bbox_pred(self.reg_net(feat)))
        
        pred_scores_list = [
            transpose_to_bs_hwa_k(s, self.num_classes) for s in pred_scores
        ]
        pred_boxes_list = [
            transpose_to_bs_hwa_k(s, 4) for s in pred_boxes
        ]

        if self.training:
            anchors = paddle.concat(anchors)
            loss_dict = self.get_loss(anchors, [pred_scores_list, pred_boxes_list], inputs)

            return loss_dict
        
        else:
            img_whwh = paddle.concat([inputs["imgs_shape"][:, 1:2],
                                      inputs["imgs_shape"][:, 0:1]], axis=-1)
            pred_result, bbox_num = self.postprocess(
                pred_scores_list, 
                pred_boxes_list, 
                anchors,
                inputs["scale_factor_wh"], 
                img_whwh
            )

            return pred_result, bbox_num


def transpose_to_bs_hwa_k(tensor, k):
    assert tensor.dim() == 4
    bs, _, h, w = tensor.shape
    tensor = tensor.reshape([bs, -1, k, h, w])
    tensor = tensor.transpose([0, 3, 4, 1, 2])

    return tensor.reshape([bs, -1, k])
