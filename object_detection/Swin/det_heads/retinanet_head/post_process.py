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

import paddle
import paddle.nn.functional as F

from det_utils.box_utils import nonempty_bbox, delta2bbox, multiclass_nms

class RetinaNetPostProcess(object):
    '''
    This class used to post_process the RetianNet-Head's output.
    '''
    def __init__(self, 
                 score_threshold,
                 keep_top_k,
                 nms_top_k,
                 nms_threshold,
                 bbox_reg_weights=[1.0, 1.0, 1.0, 1.0]):
        super(RetinaNetPostProcess, self).__init__()
        self.score_threshold=score_threshold
        self.keep_topk=keep_top_k
        self.topk_candidates=nms_top_k
        self.num_thresh=nms_threshold
        self.bbox_reg_weights = bbox_reg_weights

    def _process_single_level_pred(self, box_lvl, score_lvl, anchors, scale_factor_wh, img_whwh):
        if isinstance(scale_factor_wh, list):
            scale_factor_wh = paddle.concat(scale_factor_wh)
        if isinstance(img_whwh, list):
            img_whwh = paddle.concat(img_whwh)

        score_lvl = paddle.transpose(score_lvl, [0, 2, 1])
        score_lvl = F.sigmoid(score_lvl)

        batch_lvl = []
        for i in range(len(img_whwh)):
            box_lvl_i = delta2bbox(box_lvl[i],
                                    anchors,
                                    self.bbox_reg_weights).reshape(anchors.shape)

            box_lvl_i[:, 0::2] = paddle.clip(
                box_lvl_i[:, 0::2], min=0, max=img_whwh[i][0]
            ) / scale_factor_wh[i][0]
            box_lvl_i[:, 1::2] =  paddle.clip(
                box_lvl_i[:, 1::2], min=0, max=img_whwh[i][1]
            ) / scale_factor_wh[i][1]

            batch_lvl.append(box_lvl_i)

        box_lvl = paddle.stack(batch_lvl)

        return box_lvl, score_lvl

    def __call__(self, pred_scores_list, pred_boxes_list, anchors, scale_factor_wh, img_whwh):
        """
        Args:
            pred_scores_list (list[Tensor]): tensor of shape (batch_size, R, num_classes).
                The tensor predicts the classification probability for each proposal.
            pred_boxes_list (list[Tensor]): tensors of shape (batch_size, R, 4).
                The tensor predicts anchor's delta
            anchors (list[Tensor]): mutil-level anchors.
            scale_factor_wh (Tensor): tensors of shape [batch_size, 2] the scalor of  per img
            img_whwh (Tensor): tensors of shape [batch_size, 4]
        Returns:
            bbox_pred (Tensor): tensors of shape [num_boxes, 6] Each row has 6 values:
            [label, confidence, xmin, ymin, xmax, ymax]
            bbox_num (Tensor): tensors of shape [batch_size] the number of RoIs in each image.
        """
        assert len(pred_boxes_list[0]) == len(scale_factor_wh) == len(img_whwh)
        assert len(pred_boxes_list) == len(anchors)

        mutil_level_bbox = []
        mutil_level_score = []

        for i in range(len(pred_boxes_list)):
            lvl_res_b, lvl_res_s = self._process_single_level_pred(
                pred_boxes_list[i],
                pred_scores_list[i],
                anchors[i],
                scale_factor_wh,
                img_whwh)

            mutil_level_bbox.append(lvl_res_b)
            mutil_level_score.append(lvl_res_s)

        pred_boxes = paddle.concat(mutil_level_bbox, axis=1)     # [N, \sum_{i=0}^{n} (Hi * Wi), 4]
        pred_scores = paddle.concat(mutil_level_score, axis=2)

        assert pred_boxes.shape[1] == pred_scores.shape[2]

        bbox_pred, bbox_num, _ = multiclass_nms(
            pred_boxes, 
            pred_scores,
            score_threshold=self.score_threshold,
            keep_top_k=self.keep_topk,
            nms_top_k=self.topk_candidates,
            nms_threshold=self.num_thresh,
        )

        pred_label = bbox_pred[:, 0:1]
        pred_score = bbox_pred[:, 1:2]
        pred_bbox = bbox_pred[:, 2:]
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = paddle.unsqueeze(keep_mask, [1])
        pred_label = paddle.where(keep_mask, pred_label,
                                  paddle.ones_like(pred_label) * -1)

        pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)

        return pred_result, bbox_num
