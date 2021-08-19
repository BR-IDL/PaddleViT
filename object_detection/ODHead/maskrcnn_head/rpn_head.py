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
from paddle.nn.initializer import Normal

import sys
sys.path.append("PPViT-od_head/object_detection/head")
from det_utils.generator_utils import AnchorGenerator, ProposalGenerator
from det_utils.target_assign import anchor_target_assign


class RPNHead(nn.Layer):
    """
    Region Proposal Network uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv 
    predicts objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas.

    Attributes:
        anchor_generator (class): the generator of anchor. 
        train_proposal (class): configure of proposals generation at the stage of training.
        test_proposal (class): configure of proposals generation at the stage of prediction.
        in_channels (int): channel of input feature maps which can be derived by from_config.
    """
    def __init__(self, config):
        super(RPNHead, self).__init__()
        self.anchor_generator = AnchorGenerator(anchor_sizes=config.RPN.ANCHOR_SIZE,
                                                aspect_ratios=config.RPN.ASPECT_RATIOS,
                                                strides=config.RPN.STRIDES,
                                                offset=config.RPN.OFFSET)
        self.train_proposal = ProposalGenerator(pre_nms_top_n=config.RPN.PRE_NMS_TOP_N_TRAIN,
                                                post_nms_top_n=config.RPN.POST_NMS_TOP_N_TRAIN,
                                                nms_thresh=config.RPN.NMS_THRESH,
                                                min_size=config.RPN.MIN_SIZE,
                                                topk_after_collect=config.RPN.TOPK_AFTER_COLLECT)
        self.test_proposal = ProposalGenerator(pre_nms_top_n=config.RPN.PRE_NMS_TOP_N_TEST,
                                               post_nms_top_n=config.RPN.POST_NMS_TOP_N_TEST,
                                               nms_thresh=config.RPN.NMS_THRESH,
                                               min_size=config.RPN.MIN_SIZE,
                                               topk_after_collect=config.RPN.TOPK_AFTER_COLLECT)

        self.num_anchors = self.anchor_generator.num_anchors

        # Each level must have the same channel
        # num_channels = [feat.shape[1] for feat in feats]
        # assert len(set(num_channels)) == 1, "Each level must have the same channel!"
        # num_channels = num_channels[0]

        num_channels = config.FPN.OUT_CHANNELS
        self.conv = nn.Conv2D(num_channels,
                              num_channels,
                              kernel_size=3,
                              padding=1,
                              weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))

        self.objectness_logits = nn.Conv2D(num_channels,
                                           self.num_anchors,
                                           kernel_size=1,
                                           padding=0,
                                           weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))

        self.anchor_deltas = nn.Conv2D(num_channels,
                                       self.num_anchors * 4,
                                       kernel_size=1,
                                       padding=0,
                                       weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))

        self.config = config

    def predict(self, feats):
        '''
        Predict the logits of each feature and the deltas of the anchors in each feature.

        Args:
            feats (list[tensor]): Mutil-level feature from fpn.

        Returns:
            pred_objectness_logits (list[tensor]): A list of L elements.Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            pred_anchor_deltas (list[tensor]): A list of L elements. Element i is a tensor of shape (N, A * 4, Hi, Wi) 
                representing the predicted "deltas" used to transform anchors to proposals.
        '''
        
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for feat in feats:
            out = F.relu(self.conv(feat))
            pred_objectness_logits.append(self.objectness_logits(out))
            pred_anchor_deltas.append(self.anchor_deltas(out))

        return pred_objectness_logits, pred_anchor_deltas
    
    def _get_proposals(self, scores, bbox_deltas, anchors, inputs):
        '''
        Args:
            scores (list[tensor]): the prediction logits of the mutil-level features.
                scores[i].shape is [N, A, Hi, Wi]
            bbox_deltas (list[tensor]): the prediction anchor deltas of the mutil-level features.
                bbox_deltas[i].shape is [N, 4 * A, Hi, Wi]
            anchors (list[tensor]): the prediction anchor of the mutil-level features.
                anchors[i].shape is [Hi * Wi * A, 4]
            inputs (dict): ground truth info.
        '''
        proposal_gen = self.train_proposal if self.training else self.test_proposal
        imgs_shape = inputs["imgs_shape"]
        batch_size = imgs_shape.shape[0]

        batch_proposal_rois = []
        batch_proposal_rois_num = []
        for i in range(batch_size):
            single_img_rois_list = []
            single_img_prob_list = []

            for level_scores, level_deltas, level_anchors in zip(scores, bbox_deltas, anchors):
                level_rois, level_rois_prob, _, post_nms_top_n = proposal_gen(
                    scores = level_scores[i:i+1],
                    bbox_deltas = level_deltas[i:i+1],
                    anchors = level_anchors,
                    imgs_shape = imgs_shape[i:i+1]
                    )
                if level_rois.shape[0] > 0:
                    single_img_rois_list.append(level_rois)
                    single_img_prob_list.append(level_rois_prob)
            
            if len(single_img_rois_list) == 0:
                single_img_rois = paddle.zeros(shape=[0, 4]).astype("float32")
            else:
                single_img_rois = paddle.concat(single_img_rois_list)
                single_img_prob = paddle.concat(single_img_prob_list).flatten()

                if single_img_prob.shape[0] > post_nms_top_n:
                    single_img_topk_prob, topk_inds = paddle.topk(single_img_prob, post_nms_top_n)
                    single_img_topk_rois = paddle.gather(single_img_rois, topk_inds)
                else:
                    single_img_topk_rois = single_img_rois
            
            batch_proposal_rois.append(single_img_topk_rois)
            batch_proposal_rois_num.append(single_img_topk_rois.shape[0])
        
        return batch_proposal_rois, batch_proposal_rois_num
    
    def _get_losses(self, pred_logits, pred_loc, anchors, inputs):
        anchors = paddle.concat(anchors)
        gt_boxes = inputs["gt_boxes"]
        is_crowd = inputs.get("is_crowd", None)

        tgt_scores, tgt_bboxes, tgt_deltas = anchor_target_assign(
            anchors,
            gt_boxes,
            positive_thresh = self.config.RPN.POSITIVE_THRESH,
            negative_thresh = self.config.RPN.NEGATIVE_THRESH,
            batch_size_per_image = self.config.RPN.BATCH_SIZE_PER_IMG,
            positive_fraction = self.config.RPN.POSITIVE_FRACTION,
            allow_low_quality_matches = self.config.RPN.LOW_QUALITY_MATCHES,
            is_crowd = is_crowd
            )

        # reshape to [N, Hi * Wi * A, 1] for compute loss
        pred_scores = [
            s.transpose([0, 2, 3, 1]).reshape([s.shape[0], -1, 1]) for s in pred_logits
            ]
        
        pred_deltas = [
            d.transpose([0, 2, 3, 1]).reshape([d.shape[0], -1, 4]) for d in pred_loc
        ]

        pred_scores = paddle.concat(pred_scores, axis = 1).reshape([-1])
        pred_deltas = paddle.concat(pred_deltas, axis = 1).reshape([-1, 4])

        tgt_scores = paddle.concat(tgt_scores).astype("float32")
        tgt_deltas = paddle.concat(tgt_deltas).astype("float32")
        tgt_scores.stop_gradient = True
        tgt_deltas.stop_gradient = True

        pos_idx = paddle.nonzero(tgt_scores == 1)
        valid_idx = paddle.nonzero(tgt_scores >= 0)

        if valid_idx.shape[0] == 0:
            loss_rpn_cls = paddle.zeros([1]).astype("float32")
        else:
            pred_scores = paddle.gather(pred_scores, valid_idx)
            tgt_scores = paddle.gather(tgt_scores, valid_idx).astype("float32")
            tgt_scores.stop_gradient = True
            loss_rpn_cls = F.binary_cross_entropy_with_logits(
                logit=pred_scores, 
                label=tgt_scores, 
                reduction="sum"
                )

        if pos_idx.shape[0] == 0:
            loss_rpn_reg = paddle.zeros([1]).astype("float32")
        else:
            pred_deltas = paddle.gather(pred_deltas, pos_idx)
            tgt_deltas = paddle.gather(tgt_deltas, pos_idx)
            loss_rpn_reg = paddle.abs(pred_deltas - tgt_deltas).sum()

        norm = self.config.RPN.BATCH_SIZE_PER_IMG * len(gt_boxes)

        return {
            'loss_rpn_cls': loss_rpn_cls / norm,
            'loss_rpn_reg': loss_rpn_reg / norm
        }

    def forward(self, feats, inputs):
        '''
        Args:
            feats (list[tensor]): Mutil-level feature from fpn.
            inputs (dict): ground truth info.
        
        Returns:
            rois (list[tensor]): rois[i] is proposals of the i'th img.
            rois_num (list[int]): rois[i] is number of the i'th img's proposals. 
            losses_dict (dict | None): when training is dict contains loss_rpn_cls and loss_rpn_reg.
        '''
        pred_objectness_logits, pred_anchor_deltas = self.predict(feats)
        anchors = self.anchor_generator(feats)

        rois, rois_num = self._get_proposals(pred_objectness_logits, pred_anchor_deltas, anchors, inputs)
        
        if self.training:
            losses_dict = self._get_losses(pred_objectness_logits, pred_anchor_deltas, anchors, inputs)

            return rois, rois_num, losses_dict
        else:
            return rois, rois_num, None
