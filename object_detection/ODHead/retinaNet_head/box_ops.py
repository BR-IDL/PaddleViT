import paddle

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


def nonempty_bbox(boxes, min_size=0, return_mask=False):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = paddle.logical_and(h > min_size, w > min_size)
    if return_mask:
        return mask
    keep = paddle.nonzero(mask).flatten()
    return keep

def rbox2poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = paddle.shape(rrects)[0]

    x_ctr = rrects[:, 0]
    y_ctr = rrects[:, 1]
    width = rrects[:, 2]
    height = rrects[:, 3]
    angle = rrects[:, 4]

    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

    normal_rects = paddle.stack(
        [tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y], axis=0)
    normal_rects = paddle.reshape(normal_rects, [2, 4, N])
    normal_rects = paddle.transpose(normal_rects, [2, 0, 1])

    sin, cos = paddle.sin(angle), paddle.cos(angle)
    # M.shape=[N,2,2]
    M = paddle.stack([cos, -sin, sin, cos], axis=0)
    M = paddle.reshape(M, [2, 2, N])
    M = paddle.transpose(M, [2, 0, 1])

    # polys:[N,8]
    polys = paddle.matmul(M, normal_rects)
    polys = paddle.transpose(polys, [2, 1, 0])
    polys = paddle.reshape(polys, [-1, N])
    polys = paddle.transpose(polys, [1, 0])

    tmp = paddle.stack(
        [x_ctr, y_ctr, x_ctr, y_ctr, x_ctr, y_ctr, x_ctr, y_ctr], axis=1)
    polys = polys + tmp
    return polys

def delta2bbox(deltas, boxes, weights):
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

def bbox2delta(src_boxes, tgt_boxes, weights):
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