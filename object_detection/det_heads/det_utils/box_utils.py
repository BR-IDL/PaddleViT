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
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper

def bbox2delta(src_boxes, tgt_boxes, weights=[1.0, 1.0, 1.0, 1.0]):
    '''
    The function is used to compute two tensor boxes difference among (x, y, w, h).

    Args:
        src_boxes (tensor): shape [N, 4].
        tgt_boxes (tensor): shape [N, 4].
        weights (list[float]): balance the dx, dy, dw, dh.
    
    Returns:
        deltas (tensor): shape[N, 4].
    '''
    src_w = src_boxes[:, 2] - src_boxes[:, 0]
    src_h = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_w
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_h

    tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
    tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
    tgt_ctr_x = tgt_boxes[:, 0] + 0.5 * tgt_w
    tgt_ctr_y = tgt_boxes[:, 1] + 0.5 * tgt_h

    wx, wy, ww, wh = weights
    dx = wx * (tgt_ctr_x - src_ctr_x) / src_w
    dy = wy * (tgt_ctr_y - src_ctr_y) / src_h
    dw = ww * paddle.log(tgt_w / src_w)
    dh = wh * paddle.log(tgt_h / src_h)

    deltas = paddle.stack((dx, dy, dw, dh), axis=1)
    return deltas


def delta2bbox(deltas, boxes, weights=[1.0, 1.0, 1.0, 1.0]):
    '''
    The inverse process of bbox2delta.
    '''
    clip_scale = math.log(1000.0 / 16)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh
    # Prevent sending too large values into paddle.exp()
    dw = paddle.clip(dw, max=clip_scale)
    dh = paddle.clip(dh, max=clip_scale)

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = paddle.exp(dw) * widths.unsqueeze(1)
    pred_h = paddle.exp(dh) * heights.unsqueeze(1)

    pred_boxes = []
    pred_boxes.append(pred_ctr_x - 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y - 0.5 * pred_h)
    pred_boxes.append(pred_ctr_x + 0.5 * pred_w)
    pred_boxes.append(pred_ctr_y + 0.5 * pred_h)
    pred_boxes = paddle.stack(pred_boxes, axis=-1)

    return pred_boxes


def boxes_area(boxes):
    '''
    Compute boxes area.

    Args:
        boxes (tensor):  shape [M, 4] | [N, M, 4].

    Returns:
        areas (tensor): shape [M] | [N, M].
    '''
    assert boxes.shape[-1] == 4
    if boxes.dim() == 2:
        boxes_wh = boxes[:, 2:] - boxes[:, :2]
        return (boxes_wh[:, 0] * boxes_wh[:, 1]).clip(min=0)

    elif boxes.dim() == 3:
        boxes_wh = boxes[:, :, 2:] - boxes[:, :, :2]
        return (boxes_wh[:, :, 0] * boxes_wh[:, :, 1]).clip(min=0)

    else:
        raise ValueError("The dim of boxes must be 2 or 3!")


def boxes_iou(boxes1, boxes2, mode='a', type='iou'):
    '''
    Compute the ious of two boxes tensor and the coordinate format of boxes is xyxy.

    Args:
        boxes1 (tensor): when mode == 'a': shape [M, 4];  when mode == 'b': shape [M, 4]
        boxes2 (tensor): when mode == 'a': shape [R, 4];  when mode == 'b': shape [M, 4]
        mode (string | 'a' or 'b'): when mode == 'a': compute one to many;
                                    when mode == 'b': compute one to one.

    Returns:
        ious (tensor): when mode == 'a': shape [M, R];  when mode == 'b': shape [M]
    '''
    area1 = boxes_area(boxes1)
    area2 = boxes_area(boxes2)

    if mode == 'a':
        lt = paddle.maximum(boxes1.unsqueeze(-2)[:, :, :2], boxes2.unsqueeze(0)[:, :, :2])
        rb = paddle.minimum(boxes1.unsqueeze(-2)[:, :, 2:], boxes2.unsqueeze(0)[:, :, 2:])

        inter_wh = (rb - lt).astype("float32").clip(min=0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        union_area = area1.unsqueeze(-1) + area2 - inter_area + 1e-6

        ious = paddle.where(inter_area > 0,
                            inter_area / union_area,
                            paddle.zeros_like(inter_area, dtype="float32"))

        if type == "giou":
            enclosing_lt = paddle.maximum(boxes1.unsqueeze(-2)[:, :, :2], boxes2.unsqueeze(0)[:, :, :2])
            enclosing_rb = paddle.minimum(boxes1.unsqueeze(-2)[:, :, 2:], boxes2.unsqueeze(0)[:, :, 2:])

            enclosing_wh = (enclosing_rb - enclosing_lt).astype("float32").clip(min=0)
            enclosing_area = enclosing_wh[:, 0] * enclosing_wh[:, 1] + 1e-6

            giou = ious - (enclosing_area - union_area) / enclosing_area

            return giou, union_area

        return ious, union_area

    elif mode == 'b':
        assert boxes1.shape[0] == boxes2.shape[0]

        lt = paddle.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = paddle.minimum(boxes1[:, 2:], boxes2[:, 2:])

        inter_wh = (rb - lt).astype("float32").clip(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        union_area = area1 + area2 - inter_area + 1e-6

        ious = paddle.where(inter_area > 0,
                            inter_area / union_area,
                            paddle.zeros_like(inter_area, dtype="float32"))

        if type == "giou":
            enclosing_lt = paddle.minimum(boxes1[:, :2], boxes2[:, :2])
            enclosing_rb = paddle.maximum(boxes1[:, 2:], boxes2[:, 2:])

            enclosing_wh = (enclosing_rb - enclosing_lt).astype("float32").clip(min=0)
            enclosing_area = enclosing_wh[:, 0] * enclosing_wh[:, 1] + 1e-6

            giou = ious - (enclosing_area - union_area) / enclosing_area

            return giou, union_area

        return ious, union_area
        
    else:
        raise ValueError("Only support mode 'a' or 'b'")


def batch_iou(boxes1, boxes2, mode='a'):
    '''
    Compute the ious of two boxes tensor and the coordinate format of boxes is xyxy.

    Args:
        boxes1 (tensor): when mode == 'a': shape [N, M, 4];  when mode == 'b': shape [N, M, 4]
        boxes2 (tensor): when mode == 'a': shape [N, R, 4];  when mode == 'b': shape [N, M, 4]
        mode (string | 'a' or 'b'): when mode == 'a': compute one to many;
        when mode == 'b': compute one to one

    Returns:
        ious (tensor): when mode == 'a': shape [N, M, R];  when mode == 'b': shape [N, M]
    '''
    area1 = boxes_area(boxes1)
    area2 = boxes_area(boxes2)

    if mode == 'a':
        lt = paddle.maximum(boxes1.unsqueeze(-2)[:, :, :, :2], boxes2.unsqueeze(1)[:, :, :, :2])
        rb = paddle.minimum(boxes1.unsqueeze(-2)[:, :, :, 2:], boxes2.unsqueeze(1)[:, :, :, 2:])

        inter_wh = (rb - lt).astype("float32").clip(min=0)
        inter_area = inter_wh[:, :, :, 0] * inter_wh[:, :, :, 1]

        union_area = area1.unsqueeze(-1) + area2.unsqueeze(-2) - inter_area + 1e-6

        ious = paddle.where(inter_area > 0,
                            inter_area / union_area,
                            paddle.zeros_like(inter_area, dtype="float32"))

        return ious, union_area

    elif mode == 'b':
        assert boxes1.shape[0] == boxes2.shape[0]

        lt = paddle.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
        rb = paddle.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])

        inter_wh = (rb - lt).astype("float32").clip(min=0)
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

        union_area = area1 + area2 - inter_area + 1e-6

        ious = paddle.where(inter_area > 0,
                            inter_area / union_area,
                            paddle.zeros_like(inter_area, dtype="float32"))

        return ious, union_area
    else:
        raise ValueError("Only support mode 'a' or 'b'")


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = paddle.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    keep = paddle.nonzero(mask).flatten()
    return keep


def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   keep_top_k,
                   nms_top_k=-1,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (tensor): Two types of bboxes are supported:
                           1. (tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (tensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (tensor): Two types of scores are supported:
                           1. (tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.

    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)

        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               rois_num, *attrs)
        if not return_index:
            index = None

        return output, nms_rois_num, index