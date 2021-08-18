import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .box_ops import nonempty_bbox, rbox2poly, delta2bbox

class RetinaNetPostProcess(object):
    __inject__ = ['nms']

    def __init__(self, 
                 nms,
                 bbox_reg_weights=[1.0, 1.0, 1.0, 1.0]):

        super(RetinaNetPostProcess, self).__init__()
        self.nms = nms
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

        bbox_pred, bbox_num, _ = self.nms(pred_boxes, pred_scores)

        pred_label = bbox_pred[:, 0:1]
        pred_score = bbox_pred[:, 1:2]
        pred_bbox = bbox_pred[:, 2:]
        keep_mask = nonempty_bbox(pred_bbox, return_mask=True)
        keep_mask = paddle.unsqueeze(keep_mask, [1])
        pred_label = paddle.where(keep_mask, pred_label,
                                  paddle.ones_like(pred_label) * -1)
                                  
        pred_result = paddle.concat([pred_label, pred_score, pred_bbox], axis=1)
        
        return pred_result, bbox_num