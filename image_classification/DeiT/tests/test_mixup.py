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
import paddle
import paddle.nn as nn
from mixup import *


class MixupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')

    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_rand_bbox(self):
        image_shape = [4, 3, 224, 224] 
        lam = 0.2
        cut_rate = np.sqrt(1. - lam)
        for i in range(20):
            x1, y1, x2, y2 = rand_bbox(image_shape, lam)
            #print(x1, y1, x2, y2)
            h = x2 - x1
            w = y2 - y1
            self.assertTrue(0 <= x1 <= 224)
            self.assertTrue(0 <= y1 <= 224)
            self.assertTrue(0 <= x2 <= 224)
            self.assertTrue(0 <= y2 <= 224)
            self.assertTrue(h <= int(cut_rate * 224))
            self.assertTrue(w <= int(cut_rate * 224))

    def test_rand_bbox_minmax(self):
        image_shape = [4, 3, 224, 224] 
        minmax = [0.1, 0.3]
        for i in range(20):
            x1, y1, x2, y2 = rand_bbox_minmax(image_shape, minmax)
            h = x2 - x1
            w = y2 - y1
            self.assertTrue(0 <= x1 <= 224)
            self.assertTrue(0 <= y1 <= 224)
            self.assertTrue(0 <= x2 <= 224)
            self.assertTrue(0 <= y2 <= 224)
            self.assertTrue(h >= int(minmax[0]* 224))
            self.assertTrue(w >= int(minmax[0]* 224))
            self.assertTrue(h <= int(minmax[1]* 224))
            self.assertTrue(w <= int(minmax[1]* 224))

    #@unittest.skip('skip for debug')
    def test_cutmix_generate_bbox_adjust_lam_lam(self):
        image_shape = [4, 3, 224, 224] 
        orig_lam = 0.2
        cut_rate = np.sqrt(1. - orig_lam)
        minmax = None
        (x1, y1, x2, y2), lam = cutmix_generate_bbox_adjust_lam(image_shape, orig_lam, minmax)
        h = x2 - x1
        w = y2 - y1
        self.assertTrue(0 <= x1 <= 224)
        self.assertTrue(0 <= y1 <= 224)
        self.assertTrue(0 <= x2 <= 224)
        self.assertTrue(0 <= y2 <= 224)
        self.assertTrue(h <= cut_rate * 224)
        self.assertTrue(w <=cut_rate * 224)
        self.assertNotEqual(orig_lam, lam)

    #@unittest.skip('skip for debug')
    def test_cutmix_generate_bbox_adjust_lam_minmax(self):
        image_shape = [4, 3, 224, 224] 
        orig_lam = 0.2
        minmax = [0.1, 0.3]
        (x1, y1, x2, y2), lam = cutmix_generate_bbox_adjust_lam(image_shape, orig_lam, minmax)
        h = x2 - x1
        w = y2 - y1
        self.assertTrue(0 <= x1 <= 224)
        self.assertTrue(0 <= y1 <= 224)
        self.assertTrue(0 <= x2 <= 224)
        self.assertTrue(0 <= y2 <= 224)
        self.assertTrue(h >= minmax[0]* 224 - 1)
        self.assertTrue(w >= minmax[0]* 224 - 1)
        self.assertTrue(h <= minmax[1]* 224 - 1)
        self.assertTrue(w <= minmax[1]* 224 - 1)
        self.assertNotEqual(orig_lam, lam)

    #@unittest.skip('skip for debug')
    def test_one_hot(self):
        num_classes = 10
        x = paddle.randint(0, num_classes, [4])
        x_smoothed = one_hot(x, num_classes, on_value=0.8, off_value=0.2)
        for i in range(4):
            self.assertEqual(x_smoothed[i, x[i]], 0.8)
            for j in range(num_classes):
                if j != x[i]:
                    self.assertEqual(x_smoothed[i, j], 0.2)

    #@unittest.skip('skip for debug')
    def test_mixup_one_hot(self):
        num_classes = 10
        x = paddle.randint(0, num_classes, [4])
        x_mixup = mixup_one_hot(x, num_classes, lam=0.8, smoothing=0.2)
        off_value = 0.2 / 10
        on_value = 1. - 0.2 + off_value
        for i in range(4):
            if x[i] != x[-(i+1)]:
                self.assertAlmostEqual(x_mixup[i, x[i]].numpy()[0], on_value*0.8 + off_value * 0.2, places=4)
            else:
                self.assertAlmostEqual(x_mixup[i, x[i]].numpy()[0], on_value*0.8 + on_value * 0.2, places=4)

    #@unittest.skip('skip for debug')
    def test_mixup(self):
        x = paddle.randn([4, 3, 224, 224])
        label = paddle.randint(0, 10, [4])
        mixup_fn = Mixup(num_classes=10, cutmix_alpha=1.0) 
        x_new, label_new = mixup_fn(x, label)
        self.assertEqual(x_new.shape, x.shape)
        self.assertEqual(label_new.shape, [4, 10])

        mixup_fn = Mixup(num_classes=10, cutmix_alpha=0.2) 
        x_new, label_new = mixup_fn(x, label)
        self.assertEqual(x_new.shape, x.shape)
        self.assertEqual(label_new.shape, [4, 10])
