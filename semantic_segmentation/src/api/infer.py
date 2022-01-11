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

import numpy as np
import math
import cv2
import collections.abc
import paddle
import paddle.nn.functional as F

def slide_inference(model, imgs, crop_size, stride_size, num_classes):
    """
    Inference by sliding-window with overlap, the overlap is equal to stride.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list): the size of sliding window, (w, h).
        stride_size (tuple|list): the size of stride, (w, h).
        num_classes (int): the number of classes

    Return:
        final_logit (Tensor): The logit of input image, whose size is equal to 
        the size of img (not the orginal size).
    """
    batch_size = len(imgs)
    h_img = [img.shape[-2] for img in imgs]
    w_img = [img.shape[-1] for img in imgs]
    max_h, max_w = max(h_img), max(w_img)
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride_size
    rows = max(max_h - h_crop + h_stride -1, 0) // h_stride + 1
    cols = max(max_w - w_crop + w_stride -1, 0) // w_stride + 1
    count = paddle.zeros([batch_size, 1, max_h, max_w])
    final_logit = paddle.zeros([batch_size, num_classes, max_h, max_w])
    for r in range(rows):
        for c in range(cols):
            batch_list = []
            loc_list = []
            for i, img in enumerate(imgs):
                h1 = r * h_stride
                w1 = c * w_stride
                if h1 >= img.shape[-2] or w1 >= img.shape[-1]:
                    continue
                h2 = min(h1 + h_crop, img.shape[-2])
                w2 = min(w1 + w_crop, img.shape[-1])
                h1 = max(h2 - h_crop, 0)
                w1 = max(w2 - w_crop, 0)
                loc_list.append((i, h1, w1, h2, w2))
                batch_list.append(img[:, h1:h2, w1:w2].unsqueeze(0))
            if not batch_list:
                continue
            batch_data = paddle.concat(batch_list, 0)
            logits = model(batch_data)[0]
            for i in range(batch_data.shape[0]):
                idx, h1, w1, h2, w2 = loc_list[i]
                logit = logits[i]
                final_logit[idx, :, h1:h2, w1:w2] += logit[:,:,:]
                count[idx, :, h1:h2, w1:w2] += 1
    final_logit_list = []
    for i in range(batch_size):
        h, w = imgs[i].shape[-2:]
        logit = final_logit[i:i+1, :, :h, :w]
        count_single = count[i:i+1, :, :h, :w]
        final_logit_list.append(logit / count_single)
    return final_logit_list

def ss_inference(model,
                 img, 
                 ori_shape, 
                 is_slide, 
                 base_size, 
                 stride_size, 
                 crop_size, 
                 num_classes, 
                 rescale_from_ori=False):
    """
    Single-scale inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        img (Tensor): the input image.
        ori_shape (list): origin shape of image.
        is_slide (bool): whether to infer by sliding window.
        base_size (list): the size of short edge is resize to min(base_size) 
        when it is smaller than min(base_size)  
        stride_size (tuple|list): the size of stride, (w, h). It should be 
        probided if is_slide is True.
        crop_size (tuple|list). the size of sliding window, (w, h). It should 
        be probided if is_slide is True.
        num_classes (int): the number of classes
        rescale_from_ori (bool): whether rescale image from the original size. 
        Default: False.

    Returns:
        pred (tensor): If ori_shape is not None, a prediction with shape (1, 1, h, w) 
        is returned. If ori_shape is None, a logit with shape (1, num_classes, 
        h, w) is returned.
    """
    if not is_slide:
        if not isinstance(img, collections.abc.Sequence):
            raise TypeError("The type of img must be one of "
                "collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(img)))
        if len(img) == 1:
            img = img[0]
        else:
            raise ValueError("Considering the different shapes of inputs,"
                "batch_size should be set to 1 while is_slide is False")
        logits = model(img)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError("The type of logits must be one of "
                "collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        logit = logits[0]
    else:
        # TODO (wutianyiRosun@gmail.com): when dataloader does not uses resize,
        #  rescale or padding
        if rescale_from_ori:
            h, w = img.shape[-2], img.shape[-1]
            if min(h,w) < min(base_size):
                new_short = min(base_size)
                if h > w :
                    new_h, new_w = int(new_short * h / w), new_short
                else:
                    new_h, new_w = new_short, int(new_short * w / h)
                h, w = new_h, new_w
                img = F.interpolate(img, (h, w), mode='bilinear')
                #print("rescale, img.shape: ({}, {})".format(h,w))
        logit_list = slide_inference(model, img, crop_size, stride_size, num_classes)

    if ori_shape is not None:
        # resize to original shape
        pred_list = []
        for i, logit in enumerate(logit_list):
            shape = ori_shape[i]
            logit = F.interpolate(logit, shape, mode='bilinear', align_corners=False)  
            logit = F.softmax(logit, axis=1)
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            pred_list.append(pred)
        return pred_list
    else:
        return logit


def ms_inference(model,
                 img,
                 ori_shape,
                 is_slide,
                 base_size,
                 stride_size,
                 crop_size,
                 num_classes, 
                 scales=[1.0,],
                 flip_horizontal=True, 
                 flip_vertical=False,
                 rescale_from_ori=False):

    """
    Multi-scale inference.

    For each scale, the segmentation result is first generated by sliding-window
    testing with overlap. Then the segmentation result is resize to the original 
    size, followed by softmax operation. Finally, the segmenation logits of all 
    scales are averaged (+argmax) 

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        img (Tensor): the input image.
        ori_shape (list): origin shape of image.
        is_slide (bool): whether to infer by sliding wimdow. 
        base_size (list): the size of short edge is resize to min(base_size) 
        when it is smaller than min(base_size)  
        crop_size (tuple|list). the size of sliding window, (w, h). It should
        be probided if is_slide is True.
        stride_size (tuple|list). the size of stride, (w, h). It should be 
        probided if is_slide is True.
        num_classes (int): the number of classes
        scales (list):  scales for resize. Default: [1.0,].
        flip_horizontal (bool): whether to flip horizontally. Default: True
        flip_vertical (bool): whether to flip vertically. Default: False.
        rescale_from_ori (bool): whether rescale image from the original size. Default: False.

    Returns:
        Pred (tensor): Prediction of image with shape (1, 1, h, w) is returned.
    """
    if not isinstance(scales, (tuple, list)):
        raise('`scales` expects tuple/list, but received {}'.format(type(scales)))
    final_logit = 0
    if rescale_from_ori:
        if not isinstance(base_size, tuple):
            raise('base_size is not a tuple, but received {}'.format(type(tupel)))
        h_input, w_input = base_size
    else:
        h_input, w_input = img.shape[-2], img.shape[-1]
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        if rescale_from_ori:
            # TODO (wutianyiRosun@gmail.com): whole image testing, rescale 
            # original image according to the scale_factor between the 
            # origianl size and scale
            # scale_factor := min ( max(scale) / max(ori_size), min(scale) / min(ori_size) ) 
            h_ori, w_ori = img.shape[-2], img.shape[-1]
            max_long_edge = max(h, w)
            max_short_edge = min(h, w)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
            # compute new size
            new_h = int(h_ori * float(scale_factor) + 0.5)
            new_w = int(w_ori * float(scale_factor) + 0.5)
            h, w = new_h, new_w
            img = F.interpolate(img, (h, w), mode='bilinear')
            logits = model(img)
            logit = logits[0]
        else:
            # sliding-window testing
            # if min(h,w) is smaller than crop_size[0], the smaller edge of the
            # image will be matched to crop_size[0] maintaining the aspect ratio
            if min(h,w) < crop_size[0]:
                new_short = crop_size[0]
                if h > w :
                    new_h, new_w = int(new_short * h / w), new_short
                else:
                    new_h, new_w = new_short, int(new_short * w / h)
                h, w = new_h, new_w
            img = F.interpolate(img, (h, w), mode='bilinear')
            logit = slide_inference(model, img, crop_size, stride_size, num_classes)

        logit = F.interpolate(logit, ori_shape, mode='bilinear', align_corners=False)  
        logit = F.softmax(logit, axis=1)
        final_logit = final_logit + logit
        # flip_horizontal testing
        if flip_horizontal == True:
            img_flip = img[:, :, :, ::-1]
            logit_flip = slide_inference(model, img_flip, crop_size, 
                stride_size, num_classes)
            logit = logit_flip[:, :, :, ::-1]
            logit = F.interpolate(logit, ori_shape, mode='bilinear', align_corners=False)  
            logit = F.softmax(logit, axis=1)
            final_logit = final_logit + logit
        # TODO (wutianyiRosun@gmail.com): add flip_vertical testing
    pred = paddle.argmax(final_logit, axis=1, keepdim=True, dtype='int32')
    return pred
