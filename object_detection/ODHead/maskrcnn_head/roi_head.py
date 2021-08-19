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
from paddle.nn.initializer import XavierNormal, XavierUniform, Normal

from det_utils.target_assign import roi_target_assign
from det_utils.generator_utils import RoIAlign
from det_utils.box_utils import bbox2delta, delta2bbox, multiclass_nms


class BoxHead(nn.Layer):
    """
    A head with several 3x3 conv layers (each followed by norm & relu), then
    several fc layers (each followed by relu) and followed by two linear layers 
    for predicting Fast R-CNN outputs.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        output_size,
        num_conv,
        conv_dim,
        num_fc,
        fc_dim,
    ):
        '''
        Attributes:
            num_classes (int): the number of class.
            in_channels (int): the channels of inputs.
            output_size (int): the size of output from pooler.
            num_conv (int): the number of conv.
            conv_dim (int): the output channels of each conv.
            num_fc (int): the number of fc.
            fc_dim (int): the output channels of each fc. 
        '''
        
        super(BoxHead, self).__init__()
        conv_dims = [conv_dim] * num_conv
        fc_dims = [fc_dim] * num_fc
        self.forward_net = nn.Sequential()

        for i, channel in enumerate(conv_dims):
            conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=channel,
                kernel_size=3,
                padding=1,
                weight_attr=paddle.ParamAttr(initializer=XavierNormal(fan_in=0.0)),
                bias_attr=True
            )

            self.forward_net.add_sublayer("conv{}".format(i), conv)
            self.forward_net.add_sublayer("act_c{}".format(i), nn.ReLU())
            in_channels = channel
        
        in_dim = output_size * output_size *in_channels
        for i, out_dim in enumerate(fc_dims):
            if i == 0:
                self.forward_net.add_sublayer("flatten", nn.Flatten())

            fc = nn.Linear(in_dim,
                           out_dim,
                           weight_attr=paddle.ParamAttr(initializer=XavierUniform(fan_in=in_dim, fan_out=in_dim)))

            self.forward_net.add_sublayer("linear{}".format(i), fc)
            self.forward_net.add_sublayer("act_f{}".format(i), nn.ReLU())
            in_dim = out_dim

        self.cls_fc = nn.Linear(in_dim, 
                                num_classes + 1,
                                weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.01)))

        self.reg_fc = nn.Linear(in_dim, 
                                num_classes * 4,
                                weight_attr=paddle.ParamAttr(initializer=Normal(mean=0., std=0.001)))

    def forward(self, inputs):
        feats = self.forward_net(inputs)
        pred_scores = self.cls_fc(feats)
        pred_deltas = self.reg_fc(feats)

        return pred_scores, pred_deltas


class RoIHead(nn.Layer):
    '''
    RoIHead will match proposals from RPNHead with gt (when training),
    crop the regions and extract per-region features using proposals,
    and make per-region predictions.
    '''
    def __init__(self, config):
        super(RoIHead, self).__init__()
        self.config = config
        self.pooler = RoIAlign(
            output_size=config.ROI.ALIGN_OUTPUT_SIZE,
            scales=config.ROI.SCALES,
            sampling_ratio=config.ROI.SAMPLING_RATIO,
            canonical_box_size=config.ROI.CANONICAL_BOX_SIZE,
            canonical_level=config.ROI.CANONICAL_LEVEL,
            min_level=config.ROI.MIN_LEVEL,
            max_level=config.ROI.MAX_LEVEL,
            aligned=config.ROI.ALIGNED
        )

        self.predictor = BoxHead(
            num_classes=config.ROI.NUM_ClASSES,
            in_channels=config.FPN.OUT_CHANNELS,
            output_size=config.ROI.ALIGN_OUTPUT_SIZE,
            num_conv=config.ROI.BOX_HEAD.NUM_CONV,
            conv_dim=config.ROI.BOX_HEAD.CONV_DIM,
            num_fc=config.ROI.BOX_HEAD.NUM_FC,
            fc_dim=config.ROI.BOX_HEAD.FC_DIM
        )
    
    def _det_forward(self, feats, proposals_info):
        roi_feats = self.pooler(feats, 
                                proposals_info["proposals"], 
                                proposals_info["num_proposals"])

        predictions = self.predictor(roi_feats)

        return predictions
    
    def _get_loss(self, preds, proposals_info):
        '''
        Args:
            preds (list[tensor]): 
               pred_scores (tensor) shape is (num_proposals, num_cls + 1), The pred class score.
               pred_deltas (tensor) shape is (num_proposals, num_cls * 4), The pred location.
        '''
        pred_scores, pred_deltas = preds
        n_s = pred_deltas.shape[0]

        proposals = proposals_info["proposals"]
        gt_classes = paddle.concat(proposals_info["gt_classes"]).reshape([-1])
        gt_boxes = paddle.concat(proposals_info["gt_boxes"])

        if len(proposals) == 0:
            proposals = paddle.zeros(shape=[n_s, 4], dtype="float32")
            tgt_scores = paddle.full(shape=[n_s,], fill_value=-1, dtype="float32")
            tgt_boxes = paddle.zeros(shape=[n_s, 4], dtype="float32")
        else:
            proposals = paddle.concat(proposals)
            tgt_scores = gt_classes.reshape([-1, 1])
            tgt_boxes = gt_boxes.reshape([-1, 4])

        losses = {
            "loss_cls": F.cross_entropy(pred_scores, tgt_scores.astype("int64"), reduction='mean')
        }

        fg_idx = paddle.nonzero(
            paddle.logical_and(gt_classes >= 0, gt_classes < self.config.ROI.NUM_ClASSES)
        ).flatten()

        fg_cls_base = paddle.gather(gt_classes, fg_idx)
        fg_cls_start = paddle.arange(0, self.config.ROI.NUM_ClASSES * fg_idx.shape[0], self.config.ROI.NUM_ClASSES)
        fg_cls_idx = fg_cls_base + fg_cls_start

        fg_idx.stop_gradient = True
        tgt_boxes.stop_gradient = True
        proposals.stop_gradient = True
        tgt_scores.stop_gradient = True
        fg_cls_base.stop_gradient = True
        fg_cls_start.stop_gradient = True

        pred_deltas = pred_deltas.reshape([-1, self.config.ROI.NUM_ClASSES, 4])
        pred_deltas = paddle.gather(pred_deltas, fg_idx, axis=0).reshape([-1, 4])

        pred_deltas = paddle.gather(pred_deltas, fg_cls_idx)

        tgt_boxes = paddle.gather(tgt_boxes, fg_idx)
        proposals = paddle.gather(proposals, fg_idx)

        tgt_deltas = bbox2delta(proposals, tgt_boxes, self.config.ROI.BOX_HEAD.REG_WEIGHTS)

        loss_reg = F.l1_loss(pred_deltas, tgt_deltas, reduction="sum") / max(gt_classes.numel(), 1.0)

        losses["loss_reg"] = loss_reg

        return losses
    
    def _inference(self, preds, proposals_info, inputs):
        num_proposals = proposals_info["num_proposals"]
        proposals = proposals_info["proposals"]
        proposals = paddle.concat(proposals)

        if not len(num_proposals):
            return None
        
        pred_scores, pred_deltas = preds
        # pred_bbox shape [num_proposals_all, num_classes, 4]
        pred_bbox = delta2bbox(pred_deltas, proposals).reshape([-1, self.config.ROI.NUM_ClASSES, 4])
        pred_bbox_list = paddle.split(pred_bbox, num_proposals)
        pred_scores = F.softmax(pred_scores, axis=-1)
        pred_scores_list = paddle.split(pred_scores, num_proposals)

        post_pred = []

        for i in range(len(pred_bbox_list)):
            num_p = num_proposals[i]
            img_pred_boxes = pred_bbox_list[i]
            img_pred_scores = pred_scores_list[i]
            img_hw = inputs["imgs_shape"][i]
            img_scale_factor = inputs["scale_factor_wh"][i]

            output = multiclass_nms(bboxes=img_pred_boxes,
                                    scores=img_pred_scores[:, :-1],
                                    score_threshold=self.config.ROI.SCORE_THRESH_INFER,
                                    keep_top_k=self.config.ROI.NMS_KEEP_TOPK_INFER,
                                    nms_threshold=self.config.ROI.NMS_THRESH_INFER,
                                    background_label=-1,
                                    rois_num=paddle.to_tensor([num_p]).astype("int32"))

            if output[1][0] == 0:
                post_pred.append([])
                continue

            post_label = output[0][:, 0:1]
            post_score = output[0][:, 1:2]
            post_boxes = output[0][:, 2:]

            post_boxes[:, 0::2] = paddle.clip(
                post_boxes[:, 0::2], min=0, max=img_hw[1]
            ) / img_scale_factor[0]

            post_boxes[:, 1::2] = paddle.clip(
                post_boxes[:, 1::2], min=0, max=img_hw[0]
            ) / img_scale_factor[1]

            boxes_w = post_boxes[:, 2] - post_boxes[:, 0]
            boxes_h = post_boxes[:, 3] - post_boxes[:, 1]

            keep = paddle.nonzero(paddle.logical_and(boxes_w > 0., boxes_h > 0.)).flatten()

            post_label = paddle.gather(post_label, keep)
            post_score = paddle.gather(post_score, keep)
            post_boxes = paddle.gather(post_boxes, keep)

            final_output = paddle.concat([post_label, post_score, post_boxes], axis=-1)
            post_pred.append(final_output)
        
        return post_pred

    def forward(self, feats, proposals, inputs):
        '''
        Args:
            feats (list[tensor]): the outputs of fpn.
            proposals (list[tensor]): list[i] denotes the proposals of the i'th imgs
                from rpn head.
            inputs (dict): the gt info, eg. gt_boxes, gt_classes, imgs_wh and so on.   
        
        Returns:
            losses (dict) | outputs (list[tensor]): 
                losses contains cls_losses and reg_losses.
                the shape of outputs[i] is [M, 6], M is the number of final preds,
                Each row has 6 values: [label, score, xmin, ymin, xmax, ymax]
        '''

        proposals_info = roi_target_assign(
            proposals,
            inputs["gt_boxes"],
            inputs["gt_classes"],
            self.config.ROI.NUM_ClASSES,
            self.config.ROI.POSITIVE_THRESH,
            self.config.ROI.NEGATIVE_THRESH,
            self.config.ROI.BATCH_SIZE_PER_IMG,
            self.config.ROI.POSITIVE_FRACTION,
            self.config.ROI.LOW_QUALITY_MATCHES
        )

        predictions = self._det_forward(feats, proposals_info)

        if self.training:
            losses = self._get_loss(predictions, proposals_info)

            return losses
        
        else:
            outputs = self._inference(predictions, proposals_info, inputs)

            return outputs
