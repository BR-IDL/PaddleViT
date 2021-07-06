
import collections.abc
from itertools import combinations

import numpy as np
import math
import cv2
import paddle
import paddle.nn.functional as F


def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.

    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            if isinstance(op.target_size, int):
                h= op.target_size
                w= op.target_size
            else:
                h, w = op.target_size[0], op.target_size[1]

        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['PaddingByAspectRatio']:
            reverse_list.append(('padding', (h, w)))
            ratio = w / h
            if ratio == op.aspect_ratio:
                pass
            elif ratio > op.aspect_ratio:
                h = int(w / op.aspect_ratio)
            else:
                w = int(h * op.aspect_ratio)
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='nearest'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu':
                pred = paddle.cast(pred, 'uint8')
                pred = F.interpolate(pred, (h, w), mode=mode)
                pred = paddle.cast(pred, 'int32')
            else:
                pred = F.interpolate(pred, (h, w), mode=mode)
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = max(h_im - h_crop + h_stride -1, 0) // h_stride + 1
    cols = max(w_im - w_crop + w_stride -1, 0) // w_stride + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    count = np.zeros([1, 1, h_im, w_im])
    final_logit = paddle.zeros([1, 60, h_im, w_im], dtype='float32')
    #print("h_im= {}, w_im={}".format(h_im, w_im))
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            logit = logits[0]
            final_logit += F.pad(logit, [w1, w_im - w2, h1, h_im - h2])
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    #final_logit = final_logit / count
    final_logit = final_logit.numpy() / count
    final_logit = paddle.to_tensor(final_logit)

    return final_logit


def inference(model,
              im,
              ori_shape=None,
              transforms=None,
              is_slide=False,
              stride=None,
              crop_size=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, h, w) is returned.
    """
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        logit = logits[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    if ori_shape is not None:

        # method 1
        logit = F.interpolate(logit, ori_shape, mode='bilinear', align_corners = False)  # resize to original shape
        logit = F.softmax(logit, axis=1)
        pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
        # method 2
        #pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
        #pred = reverse_transform(pred, ori_shape, transforms)
        return pred
    else:
        return logit


def aug_inference(model,
                  im,
                  ori_shape,
                  transforms,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=True,
                  stride=None,
                  crop_size=None):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) is returned.
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = 0
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im = F.interpolate(im, (h, w), mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im, flip)
            logit = inference(
                model,
                im_flip,
                is_slide=is_slide,
                crop_size=crop_size,
                stride=stride)
            logit = tensor_flip(logit, flip)
            logit = F.interpolate(logit, (h_input, w_input), mode='bilinear')

            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit

    pred = paddle.argmax(final_logit, axis=1, keepdim=True, dtype='int32')
    pred = reverse_transform(pred, ori_shape, transforms)
    return pred
