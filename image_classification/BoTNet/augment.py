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

"""Augmentation"""
""" Rand Augmentation """
# reference: RandAugment: Practical automated data augmentation with a reduced search space
# https://arxiv.org/abs/1909.13719

""" Auto Augmentation """
# reference: AutoAugment: Learning Augmentation Policies from Data
# https://arxiv.org/abs/1805.09501

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def auto_augment_policy_original():
    """25 types of augment policies in original paper"""
    policy = [
        [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    policy = [[SubPolicy(*args) for args in subpolicy] for subpolicy in policy]
    return policy


def rand_augment_policy_original(magnitude_idx):
    """
    14 types of augment policies in original paper
    Args:
        magnitude_idx: M
    """
    policy = [
        ('Posterize', 1, magnitude_idx), ('Rotate', 1, magnitude_idx),
        ('Solarize', 1, magnitude_idx), ('AutoContrast', 1, magnitude_idx),
        ('Equalize', 1, magnitude_idx), ('Contrast', 1, magnitude_idx),
        ('Color', 1, magnitude_idx), ('Invert', 1, magnitude_idx),
        ('Sharpness', 1, magnitude_idx), ('Brightness', 1, magnitude_idx),
        ('ShearX', 1, magnitude_idx), ('ShearY', 1, magnitude_idx),
        ('TranslateX', 1, magnitude_idx), ('TranslateY', 1, magnitude_idx),
    ]
    policy = [SubPolicy(*args) for args in policy]
    return policy


class AutoAugment():
    """Auto Augment
    Randomly choose a tuple of augment ops from a list of policy
    Then apply the tuple of augment ops to input image

    Examples:
        policy = auto_augment_policy_original()
        augment = AutoAugment(policy)
        transformed_image = augment(image)
    """

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, image, policy_idx=None):
        if policy_idx is None:
            policy_idx = random.randint(0, len(self.policy) - 1)

        sub_policy = self.policy[policy_idx]
        for op in sub_policy:
            image = op(image)
        return image


class RandAugment():
    """Rand Augment
    Randomly choose N augment ops from a list of K policies
    Then apply the N ops to input image

    Examples:
        policy = rand_augment_policy_original(magnitude_idx)
        augment = RandAugment(policy)
        transformed_image = augment(image)
    """

    def __init__(self, policy, num_layers):
        """
        Args:
            policy: list of SubPolicy
            num_layers: int
        """
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
    Read augment name and magnitude, apply augment with probability
    Args:
        op_name: str, augment operation name
        prob: float, if prob > random prob, apply augment
        magnitude_idx: int, index of magnitude in preset magnitude ranges
    """

    def __init__(self, op_name, prob, magnitude_idx):
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

        self.prob = prob
        self.magnitude = ranges[op_name][magnitude_idx]
        self.op = image_ops[op_name]

    def __call__(self, image):
        if self.prob > random.random():
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

