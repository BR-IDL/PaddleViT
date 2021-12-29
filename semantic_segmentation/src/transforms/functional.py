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

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage.morphology import distance_transform_edt


def normalize(img, mean, std):
    img = img.astype(np.float32, copy=False) / 255.0
    img -= mean
    img /= std
    return img

def imnormalize(img, mean, std):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std)

def imnormalize_(img, mean, std):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized. (0~255)
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def horizontal_flip(img):
    if len(img.shape) == 3:
        img = img[:, ::-1, :]
    elif len(img.shape) == 2:
        img = img[:, ::-1]
    return img

def vertical_flip(img):
    if len(img.shape) == 3:
        img = img[::-1, :, :]
    elif len(img.shape) == 2:
        img = img[::-1, :]
    return img

def brightness(img, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    img = ImageEnhance.Brightness(img).enhance(brightness_delta)
    return img

def contrast(img, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    img = ImageEnhance.Contrast(img).enhance(contrast_delta)
    return img

def saturation(img, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    img = ImageEnhance.Color(img).enhance(saturation_delta)
    return img

def hue(img, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    img = np.array(img.convert('HSV'))
    img[:, :, 0] = img[:, :, 0] + hue_delta
    img = Image.fromarray(img, mode='HSV').convert('RGB')
    return img

def rotate(img, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    img = img.rotate(int(rotate_delta))
    return img
