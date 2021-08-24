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

""" Transforms for image data and detection targets"""

import random
import numpy as np
import PIL
import paddle
import paddle.vision.transforms as T
from paddle.vision.transforms import functional as F
from random_erasing import RandomErasing
from box_ops import box_xyxy_to_cxcywh


def crop(image, target, region):
    cropped_image = T.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    target['size'] = paddle.to_tensor([h, w])
    fields = ['labels', 'area', 'iscrowd']

    if 'boxes' in target:
        boxes = target['boxes']
        max_size = paddle.to_tensor([h, w], dtype='float32')
        cropped_boxes = boxes - paddle.to_tensor([j, i, j, i], dtype='float32')
        cropped_boxes = paddle.minimum(cropped_boxes.reshape([-1, 2, 2]), max_size)
        cropped_boxes = cropped_boxes.clip(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
        target['boxes'] = cropped_boxes.reshape([-1, 4])
        target['area'] = area
        fields.append('boxes')

    if 'masks' in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append('masks')

    if 'boxes' in target or 'masks' in target:
        if 'boxes' in target:
            cropped_boxes = target['boxes'].reshape((-1, 2, 2))
            keep = paddle.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        keep = keep.cpu().numpy()
        keep_idx = np.where(keep)[0]
        keep = paddle.to_tensor(keep_idx)
        boxes = boxes.index_select(keep, axis=0)

        for field in fields:
            target[field] = target[field].index_select(keep, axis=0)

    return cropped_image, target


def hflip(image, target):
    flipped_image = T.hflip(image)

    w, h = image.size

    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes'] # n x 4
        boxes = boxes.index_select(paddle.to_tensor([2, 1, 0, 3]), axis=1)
        boxes = boxes * paddle.to_tensor(
            [-1, 1, -1, 1], dtype='float32') + paddle.to_tensor([w, 0, w, 0], dtype='float32')
        target['boxes'] = boxes

    if 'masks' in target:
        target['masks'] = (target['masks']).flip(axis=[-1])

    return flipped_image, target


def resize(image, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        """ get new image size for rescale, aspect ratio is kept, and longer side must < max_size
        Args:
            image_size: tuple/list of image width and height
            size: length of shorter side of scaled image
            max_size: max length of longer side of scaled image
        """
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min(w, h))
            max_original_size = float(max(w, h))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = T.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        #print('id ====== ', target['image_id'])
        #print('boxes ==========', boxes)
        scaled_boxes = boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target['boxes'] = scaled_boxes

    if 'area' in target:
        area = target['area']
        scaled_area = area * (ratio_width * ratio_height)
        target['area'] = scaled_area

    h, w = size
    target['size'] = paddle.to_tensor([h, w])

    if 'masks' in target:
        masks = target['masks'] # [N, H, W]
        masks = masks.unsqueeze(-1).astype('float32') #[N, H, W, 1]
        masks = paddle.nn.functional.interpolate(
                    masks, size, data_format='NHWC')  #[N, H', W', 1]
        masks = masks[:, :, :, 0] > 0.5
        masks = masks.astype('int32')
        target['masks'] = masks

    return rescaled_image, target


def pad(image, target, padding):
    padded_image = T.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target['size'] = paddle.to_tensor(padded_image.size[::-1])
    if 'masks' in target:
        target['masks'] = T.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop():
    def __init__(self, size):
        self.size = size
    
    @staticmethod
    def get_param(image, output_size):
        def _get_image_size(img):
            if F._is_pil_image(img):
                return img.size
            elif F._is_numpy_image(img):
                return img.shape[:2][::-1]
            elif F._is_tensor_image(img):
                return img.shape[1:][::-1]  # chw
            else:
                raise TypeError("Unexpected type {}".format(type(img)))

        w, h = _get_image_size(image)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):
        region = RandomCrop.get_param(image, self.size)
        return crop(image, target, region)


class RandomSizeCrop():
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w = random.randint(self.min_size, min(image.width, self.max_size))
        h = random.randint(self.min_size, min(image.height, self.max_size))
        region = RandomCrop.get_param(image, (h, w))
        return crop(image, target, region)


class CenterCrop():
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        image_width, image_height = image.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.)) 
        crop_left = int(round((image_width - crop_width) / 2.)) 
        return crop(image, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        if random.random() < self.p:
            return hflip(image, target)
        return image, target


class RandomResize():
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple)) 
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.choice(self.sizes)
        return resize(image, target, size, self.max_size)


class RandomPad():
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, image, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(image, target, (pad_x, pad_y))


class RandomSelect():
    """ Random select one the transforms to apply with probablity p"""
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
        
    def __call__(self, image, target):
        if random.random() > self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class ToTensor():
    def __call__(self, image, target):
        return T.to_tensor(image), target


class RandomErasing():
    def __init__(self, *args, **kwargs):
        self.eraser = RandomErasing(*args, **kwargs) 

    def __call__(self, image, target):
        return self.eraser(image), target


class Normalize():
    """Normalization for image and labels.

    Specifically, image is normalized with -mean and /std,
    boxes are converted to [cx, cy, w, h] format and scaled to 
    [0, 1] according to image size
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = T.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / paddle.to_tensor([w, h, w, h], dtype='float32')
            target['boxes'] = boxes
        return image, target


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
            #print(image)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string



        








