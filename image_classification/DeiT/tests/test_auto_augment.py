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

import unittest
import numpy as np
from PIL import Image
from auto_augment import *


class AutoAugmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #cls.img = Image.open('./lena.png')
        pass

    @classmethod
    def tearDown(cls):
        pass

    @unittest.skip('skip for debug')
    def test_shear_x(self):
        img = AutoAugmentTest.img
        img = shear_x(img, 0.3)
        img.save('lena_shear_x.png')

    @unittest.skip('skip for debug')
    def test_shear_y(self):
        img = AutoAugmentTest.img
        img = shear_y(img, 0.3)
        img.save('lena_shear_y_0.3.png')

    @unittest.skip('skip for debug')
    def test_translate_x_relative(self):
        img = AutoAugmentTest.img
        img = translate_x_relative(img, 0.25)
        img.save('lena_translate_x_r_0.25.png')

    @unittest.skip('skip for debug')
    def test_translate_y_relative(self):
        img = AutoAugmentTest.img
        img = translate_y_relative(img, 0.25)
        img.save('lena_translate_y_r_0.25.png')

    @unittest.skip('skip for debug')
    def test_translate_x_absolute(self):
        img = AutoAugmentTest.img
        img = translate_x_absolute(img, 150)
        img.save('lena_absolute_x_r_150.png')

    @unittest.skip('skip for debug')
    def test_translate_y_absolute(self):
        img = AutoAugmentTest.img
        img = translate_y_absolute(img, 150)
        img.save('lena_absolute_y_r_150.png')

    @unittest.skip('skip for debug')
    def test_rotate(self):
        img = AutoAugmentTest.img
        img = rotate(img, 30)
        img.save('lena_rotate_30.png')

    @unittest.skip('skip for debug')
    def test_auto_contrast(self):
        img = AutoAugmentTest.img
        img = auto_contrast(img)
        img.save('lena_auto_contrast.png')

    @unittest.skip('skip for debug')
    def test_invert(self):
        img = AutoAugmentTest.img
        img = invert(img)
        img.save('lena_invert_30.png')

    @unittest.skip('skip for debug')
    def test_equalize(self):
        img = AutoAugmentTest.img
        img = equalize(img)
        img.save('lena_equalize.png')

    @unittest.skip('skip for debug')
    def test_solarize(self):
        img = AutoAugmentTest.img
        img = solarize(img, 50)
        img.save('lena_solarize_50.png')

    @unittest.skip('skip for debug')
    def test_posterize(self):
        img = AutoAugmentTest.img
        img = posterize(img, 8)
        img.save('lena_posterize_8.png')

    @unittest.skip('skip for debug')
    def test_contrast(self):
        img = AutoAugmentTest.img
        img = contrast(img, 1.5)
        img.save('lena_contrast_1.5.png')

    @unittest.skip('skip for debug')
    def test_color(self):
        img = AutoAugmentTest.img
        img = color(img, 1.5)
        img.save('lena_color_1.5.png')

    @unittest.skip('skip for debug')
    def test_brightness(self):
        img = AutoAugmentTest.img
        img = brightness(img, 1.5)
        img.save('lena_brightness_1.5.png')

    @unittest.skip('skip for debug')
    def test_sharpness(self):
        img = AutoAugmentTest.img
        img = sharpness(img, 1.5)
        img.save('lena_sharpness_1.5.png')

    @unittest.skip('skip for debug')
    def test_subpolicy(self):
        img = AutoAugmentTest.img
        sub = SubPolicy('ShearX', 1.0, 3)
        img = sub(img)
        img.save('lena_subpolicy.png')

    @unittest.skip('skip for debug')
    def test_auto_augment(self):
        img = AutoAugmentTest.img
        for i in range(10):
            policy = auto_augment_policy_original()
            aa = AutoAugment(policy)
            img = aa(img)
            img.save(f'lena_aa_{i}.png')

    
