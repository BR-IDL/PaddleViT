# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

""" box related operations """

import numpy as np
import paddle


def box_xyxy_to_cxcywh_numpy(box):
    """convert box from top-left/bottom-right format:
    [x0, y0, x1, y1]
    to center-size format:
    [center_x, center_y, width, height]

    Args:
        box: numpy array, last_dim=4, stop-left/bottom-right format boxes
    Return:
        numpy array, last_dim=4, center-size format boxes
    """

    #x0, y0, x1, y1 = box.unbind(-1)
    x0 = box[:, 0]
    y0 = box[:, 1]
    x1 = box[:, 2]
    y1 = box[:, 3]
    xc = x0 + (x1-x0)/2
    yc = y0 + (y1-y0)/2
    w = x1 - x0
    h = y1 - y0
    return np.stack([xc, yc, w, h], axis=-1)


def box_cxcywh_to_xyxy(box):
    """convert box from center-size format:
    [center_x, center_y, width, height]
    to top-left/bottom-right format:
    [x0, y0, x1, y1]

    Args:
        box: paddle.Tensor, last_dim=4, stores center-size format boxes
    Return:
        paddle.Tensor, last_dim=4, top-left/bottom-right format boxes
    """

    x_c, y_c, w, h = box.unbind(-1)
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return paddle.stack([x0, y0, x1, y1], axis=-1)


def box_xyxy_to_cxcywh(box):
    """convert box from top-left/bottom-right format:
    [x0, y0, x1, y1]
    to center-size format:
    [center_x, center_y, width, height]

    Args:
        box: paddle.Tensor, last_dim=4, stop-left/bottom-right format boxes
    Return:
        paddle.Tensor, last_dim=4, center-size format boxes
    """

    x0, y0, x1, y1 = box.unbind(-1)
    xc = x0 + (x1-x0)/2
    yc = y0 + (y1-y0)/2
    w = x1 - x0
    h = y1 - y0
    return paddle.stack([xc, yc, w, h], axis=-1)


def box_area(boxes):
    """ compute area of a set of boxes in (x1, y1, x2, y2) format
    Args:
        boxes: paddle.Tensor, shape = Nx4, must in (x1, y1, x2, y2) format
    Return:
        areas: paddle.Tensor, N, areas of each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """compute iou of 2 sets of boxes in (x1, y1, x2, y2) format

    This method returns the iou between every pair of boxes
    in two sets of boxes.

    Args:
        boxes1: paddle.Tensor, shape=N x 4, boxes are stored in (x1, y1, x2, y2) format
        boxes2: paddle.Tensor, shape=N x 4, boxes are stored in (x1, y1, x2, y2) format
    Return:
        iou: iou ratios between each pair of boxes in boxes1 and boxes2
        union: union areas between each pair of boxes in boxes1 and boxes2
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    boxes1 = boxes1.unsqueeze(1) # N x 1 x 4
    lt = paddle.maximum(boxes1[:, :, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1.unsqueeze(1) + area2 - inter # broadcast

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Compute GIoU of each pais in boxes1 and boxes2

    GIoU = IoU - |A_c - U| / |A_c|
    where A_c is the smallest convex hull that encloses both boxes, U is the union of boxes
    Details illustrations can be found in https://giou.stanford.edu/

    Args:
        boxes1: paddle.Tensor, shape=N x 4, boxes are stored in (x1, y1, x2, y2) format
        boxes2: paddle.Tensor, shape=N x 4, boxes are stored in (x1, y1, x2, y2) format
    Return:
        giou: giou ratios between each pair of boxes in boxes1 and boxes2
    """

    iou, union = box_iou(boxes1, boxes2)

    boxes1 = boxes1.unsqueeze(1) # N x 1 x 4
    lt = paddle.minimum(boxes1[:, :, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area-union) / area


def masks_to_boxes(masks):
    """convert masks to bboxes

    Args:
        masks: paddle.Tensor, NxHxW
    Return:
        boxes: paddle.Tensor, Nx4
    """

    if masks.numel() == 0:
        return paddle.zeros((0, 4))
    h, w = masks.shape[-2:]
    y = paddle.arange(0, h, dtype='float32')
    x = paddle.arange(0, w, dtype='float32')
    y, x = paddle.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]

    #x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)
    x_min = paddle.where(masks == 0, paddle.ones_like(x_mask)*float(1e8), x_mask)
    x_min = x_min.flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    #y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    y_min = paddle.where(masks == 0, paddle.ones_like(y_mask) * float(1e8), y_mask)
    y_min = y_min.flatten(1).min(-1)[0]

    return paddle.stack([x_min, y_min, x_max, y_max], 1)
