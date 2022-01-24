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
                 num_classes,
                 score_threshold,
                 keep_top_k,
                 nms_top_k,
                 nms_threshold,
                 bbox_reg_weights=[1.0, 1.0, 1.0, 1.0]):

        super(RetinaNetPostProcess, self).__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.keep_topk = keep_top_k
        self.topk_candidates = nms_top_k
        self.num_thresh = nms_threshold
        self.bbox_reg_weights = bbox_reg_weights

    def transpose_to_bs_hwa_k(self, tensor, k):
        bs, _, h, w = tensor.shape
        tensor = paddle.reshape(tensor, [bs, -1, k, h, w])
        tensor = paddle.transpose(tensor, [0, 3, 4, 1, 2])

        return paddle.reshape(tensor, [bs, -1, k])

    def _process_single_level_pred(self, box_lvl, score_lvl, anchors, scale_factor_wh, img_wh):    
        score_lvl = paddle.transpose(score_lvl, [0, 2, 1])
        score_lvl = F.sigmoid(score_lvl)

        batch_lvl = []

        for i in range(box_lvl.shape[0]):
            box_lvl_i = delta2bbox(box_lvl[i], anchors, self.bbox_reg_weights)
            # don't set anchors.shape instaed of [anchors.shape[0], 4], otherwise error will be raised during dynamic to static
            # the shape of box_lvl_i shape will be [-1, -1] instead of [-1, 4], the shape of bbox_pred will be [-1, 1]
            box_lvl_i = paddle.reshape(box_lvl_i, [-1, 4])  

            box_lvl_i[:, 0::2] = paddle.clip(
                box_lvl_i[:, 0::2], min=0, max=img_wh[i][0]
            ) / scale_factor_wh[i][0]
            box_lvl_i[:, 1::2] =  paddle.clip(
                box_lvl_i[:, 1::2], min=0, max=img_wh[i][1]
            ) / scale_factor_wh[i][1]
            batch_lvl.append(box_lvl_i)

        box_lvl = paddle.stack(batch_lvl, axis=0)
        return box_lvl, score_lvl

    def __call__(self, pred_scores, pred_boxes, anchors, inputs):
        """
        Args:
            pred_scores (list[Tensor]): tensor of shape (batch_size, 9*num_classes, h, w).
                The tensor predicts the classification probability for each proposal.
            pred_boxes (list[Tensor]): tensors of shape (batch_size, 9*4, h, w).
                The tensor predicts anchor's delta
            anchors (list[Tensor]): mutil-level anchors.
            inputs (dict): groundtruth infomation.
        Returns:
            bbox_pred (Tensor): tensors of shape [num_boxes, 6] Each row has 6 values:
            [label, confidence, xmin, ymin, xmax, ymax]
            bbox_num (Tensor): tensors of shape [batch_size] the number of RoIs in each image.
        """
        if len(pred_scores[0].shape) > 3:
            pred_scores_list = [
                self.transpose_to_bs_hwa_k(s, self.num_classes) for s in pred_scores
            ]
            pred_boxes_list = [
                self.transpose_to_bs_hwa_k(s, 4) for s in pred_boxes
            ]
        else:
            pred_scores_list = [s for s in pred_scores]
            pred_boxes_list = [s for s in pred_boxes]

        if isinstance(inputs["imgs_shape"], list):
            inputs["imgs_shape"] = paddle.concat(inputs["imgs_shape"])
        
        if isinstance(inputs["scale_factor_wh"], list):
            scale_factor_wh = paddle.concat(inputs["scale_factor_wh"])
        else:
            scale_factor_wh = inputs["scale_factor_wh"]

        img_wh = paddle.concat([inputs["imgs_shape"][:, 1:2],
                                inputs["imgs_shape"][:, 0:1]],axis=-1)

        assert len(pred_boxes_list[0]) == len(scale_factor_wh) == len(img_wh)
        assert len(pred_boxes_list) == len(anchors)

        mutil_level_bbox = []
        mutil_level_score = []

        for i in range(len(pred_boxes_list)):
            lvl_res_b, lvl_res_s = self._process_single_level_pred(
                pred_boxes_list[i], 
                pred_scores_list[i], 
                anchors[i],
                scale_factor_wh, 
                img_wh
            )

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

        if bbox_num.sum() == 0:
            return bbox_pred, bbox_num

        pred_label = bbox_pred[:, 0:1]
        pred_score = bbox_pred[:, 1:2]
        pred_bbox = bbox_pred[:, 2:]

        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = paddle.unsqueeze(keep_mask, [1])
        pred_label = paddle.where(keep_mask, pred_label,
                                  paddle.ones_like(pred_label) * -1)
                                  
        pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)

        return pred_result, bbox_num
