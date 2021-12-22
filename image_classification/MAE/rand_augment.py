# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

"""Rand Augmentation

reference: RandAugment: Practical automated data augmentation with a reduced search space
https://arxiv.org/abs/1909.13719
"""


import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def rand_augment_policy_original(magnitude_idx):
    """14 types augment policy"""
    policy = [
        ('Posterize', magnitude_idx), ('Rotate', magnitude_idx),
        ('Solarize', magnitude_idx), ('AutoContrast', magnitude_idx),
        ('Equalize', magnitude_idx), ('Contrast', magnitude_idx),
        ('Color', magnitude_idx), ('Invert', magnitude_idx),
        ('Sharpness', magnitude_idx), ('Brightness', magnitude_idx),
        ('ShearX', magnitude_idx), ('ShearY', magnitude_idx),
        ('TranslateX', magnitude_idx), ('TranslateY', magnitude_idx),
    ]
    policy = [SubPolicy(*args) for args in policy]
    return policy


class RandAugment():
    """Rand Augment
    Randomly choose N ('num_layers' arg) of augment ops from a list of K policies
    Then apply the N ops to input image
    """

    def __init__(self, policy, num_layers):
        self.policy = policy
        self.num_layers = num_layers

    def __call__(self, image):
        selected_idx = np.random.choice(len(self.policy), self.num_layers)

        for policy_idx in selected_idx:
            sub_policy = self.policy[policy_idx]
            image = sub_policy(image)
        return image


class SubPolicy:
    """Subpolicy
    Read augment name and magnitude, apply augment
    Args:
        op_name: str, augment operation name
        magnitude_idx: int, index of magnitude in preset magnitude ranges
    """

    def __init__(self, op_name, magnitude_idx):
        # ranges of operations' magnitude
        ranges = {
            'ShearX': np.linspace(0, 0.3, 10),  # [-0.3, 0.3] (by random negative)
            'ShearY': np.linspace(0, 0.3, 10),  # [-0.3, 0.3] (by random negative)
            'TranslateX': np.linspace(0, 150 / 331, 10),  # [-0.45, 0.45] (by random negative)
            'TranslateY': np.linspace(0, 150 / 331, 10),  # [-0.45, 0.45] (by random negative)
            'Rotate': np.linspace(0, 30, 10),  # [-30, 30] (by random negative)
            'Color': np.linspace(0, 0.9, 10),  # [-0.9, 0.9] (by random negative)
            'Posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),  # [0, 4]
            'Solarize': np.linspace(256, 0, 10),  # [0, 256]
            'Contrast': np.linspace(0, 0.9, 10),  # [-0.9, 0.9] (by random negative)
            'Sharpness': np.linspace(0, 0.9, 10),  # [-0.9, 0.9] (by random negative)
            'Brightness': np.linspace(0, 0.9, 10),  # [-0.9, 0.9] (by random negative)
            'AutoContrast': [0] * 10,  # no range
            'Equalize': [0] * 10,  # no range
            'Invert': [0] * 10,  # no range
        }

        # augmentation operations 
        # Lambda is not pickleable for DDP
        # image_ops = {
        #    'ShearX': lambda image, magnitude: shear_x(image, magnitude),   
        #    'ShearY': lambda image, magnitude: shear_y(image, magnitude),   
        #    'TranslateX': lambda image, magnitude: translate_x(image, magnitude),   
        #    'TranslateY': lambda image, magnitude: translate_y(image, magnitude),   
        #    'Rotate': lambda image, magnitude: rotate(image, magnitude),   
        #    'AutoContrast': lambda image, magnitude: auto_contrast(image, magnitude),   
        #    'Invert': lambda image, magnitude: invert(image, magnitude),   
        #    'Equalize': lambda image, magnitude: equalize(image, magnitude),   
        #    'Solarize': lambda image, magnitude: solarize(image, magnitude),   
        #    'Posterize': lambda image, magnitude: posterize(image, magnitude),   
        #    'Contrast': lambda image, magnitude: contrast(image, magnitude),   
        #    'Color': lambda image, magnitude: color(image, magnitude),   
        #    'Brightness': lambda image, magnitude: brightness(image, magnitude),   
        #    'Sharpness': lambda image, magnitude: sharpness(image, magnitude),   
        # }
        image_ops = {
            'ShearX': shear_x,
            'ShearY': shear_y,
            'TranslateX': translate_x_relative,
            'TranslateY': translate_y_relative,
            'Rotate': rotate,
            'AutoContrast': auto_contrast,
            'Invert': invert,
            'Equalize': equalize,
            'Solarize': solarize,
            'Posterize': posterize,
            'Contrast': contrast,
            'Color': color,
            'Brightness': brightness,
            'Sharpness': sharpness,
        }

        self.magnitude = ranges[op_name][magnitude_idx]
        self.op = image_ops[op_name]

    def __call__(self, image):
        image = self.op(image, self.magnitude)
        return image


# PIL Image transforms
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform
def shear_x(image, magnitude, fillcolor=(128, 128, 128)):
    factor = magnitude * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), fillcolor=fillcolor)


def shear_y(image, magnitude, fillcolor=(128, 128, 128)):
    factor = magnitude * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), fillcolor=fillcolor)


def translate_x_relative(image, magnitude, fillcolor=(128, 128, 128)):
    pixels = magnitude * image.size[0]
    pixels = pixels * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=fillcolor)


def translate_y_relative(image, magnitude, fillcolor=(128, 128, 128)):
    pixels = magnitude * image.size[0]
    pixels = pixels * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=fillcolor)


def translate_x_absolute(image, magnitude, fillcolor=(128, 128, 128)):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor)


def translate_y_absolute(image, magnitude, fillcolor=(128, 128, 128)):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor)


def rotate(image, magnitude):
    rot = image.convert("RGBA").rotate(magnitude)
    return Image.composite(rot,
                           Image.new('RGBA', rot.size, (128,) * 4),
                           rot).convert(image.mode)


def auto_contrast(image, magnitude=None):
    return ImageOps.autocontrast(image)


def invert(image, magnitude=None):
    return ImageOps.invert(image)


def equalize(image, magnitude=None):
    return ImageOps.equalize(image)


def solarize(image, magnitude):
    return ImageOps.solarize(image, magnitude)


def posterize(image, magnitude):
    return ImageOps.posterize(image, magnitude)


def contrast(image, magnitude):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return ImageEnhance.Contrast(image).enhance(1 + magnitude)


def color(image, magnitude):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return ImageEnhance.Color(image).enhance(1 + magnitude)


def brightness(image, magnitude):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return ImageEnhance.Brightness(image).enhance(1 + magnitude)


def sharpness(image, magnitude):
    magnitude = magnitude * random.choice([-1, 1])  # random negative
    return ImageEnhance.Sharpness(image).enhance(1 + magnitude)
