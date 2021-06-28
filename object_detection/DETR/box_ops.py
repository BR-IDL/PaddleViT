import paddle

def box_cxcywh_to_xyxy(box):
    x_c, y_c, w, h = box.unbind(-1)
    x0 = x_c - 0.5 * w
    y0 = y_c - 0.5 * h
    x1 = x_c + 0.5 * w
    y1 = y_c + 0.5 * h
    return paddle.stack([x0, y0, x1, y1], axis=-1)


def box_xyxy_to_cxcywh(box):
    x0, y0, x1, y1 = box.unbind(-1)
    xc = x0 + (x1-x0)/2
    yc = y0 + (y1-y0)/2
    w = x1 - x0
    h = y1 - y0
    return paddle.stack([xc, yc, w, h], axis=-1)


def box_area(boxes):
    # compute area of a set of boxes in (x1, y1, x2, y2) format
    # in: tensor, Nx4
    # out: tensor, N
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    # compute iou of 2 sets of boxes in (x1, y1, x2, y2) format
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    boxes1 = boxes1.unsqueeze(1) # N x 1 x 4
    #print('boxes1', boxes1.shape)
    #print('boxes2', boxes2.shape)
    lt = paddle.maximum(boxes1[:, :, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1.unsqueeze(1) + area2 - inter # broadcast

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)

    boxes1 = boxes1.unsqueeze(1) # N x 1 x 4
    lt = paddle.minimum(boxes1[:, :, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, :, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area-union) / area


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return paddle.zeros((0, 4))
    h, w = masks.shape[-2:]
    y = paddle.arange(0, h, dtype=paddle.float)
    x = paddle.arange(0, w, dtype=paddle.float)
    y, x = paddle.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    #x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)
    x_max = paddle.where(masks.astype(bool) == False, paddle.ones_list(x_max) * float(1e8), x_max)
    x_min = x_max.flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    #y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    y_max = paddle.where(masks.astype(bool) == False, paddle.ones_list(y_max) * float(1e8), y_max)
    y_min = y_max.flatten(1).min(-1)[0]

    return paddle.stack([x_min, y_min, x_max, y_max], 1)


