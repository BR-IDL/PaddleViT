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
import paddle
import paddle.nn as nn
from utils import AverageMeter
from utils import WarmupCosineScheduler
from utils import get_exclude_from_weight_decay_fn


class UtilTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_average_meter(self):
        meter = AverageMeter()
        for i in range(1, 101):
            meter.update(i, 1)
        self.assertEqual(meter.avg, 50.5)

    def test_warmup_cosine_scheduler(self):
        sch = WarmupCosineScheduler(learning_rate=0.1,
                                    warmup_start_lr=1e-5,
                                    start_lr=0.1,
                                    end_lr=0.0,
                                    warmup_epochs=10,
                                    total_epochs=100,
                                    last_epoch=-1)
        lrs = []
        for epoch in range(100):
            lr = sch.get_lr()
            lrs.append(lr)
            sch.step()
        lrs.append(sch.get_lr())

        self.assertEqual(lrs[0], 1e-5)
        self.assertEqual(lrs[10], 0.1)
        self.assertEqual(lrs[-1], 0.0)
        self.assertGreaterEqual(min(lrs[0:10]), 1e-5)
        self.assertLessEqual(max(lrs[0:10]), 0.1)
        self.assertGreaterEqual(min(lrs[10::]), 0.0)
        self.assertLessEqual(max(lrs[10::]), 0.1)
            
    def test_warmup_cosine_scheduler_last_epoch(self):
        sch = WarmupCosineScheduler(learning_rate=0.1,
                                    warmup_start_lr=1e-5,
                                    start_lr=0.1,
                                    end_lr=0.0,
                                    warmup_epochs=10,
                                    total_epochs=100,
                                    last_epoch=9)
        lrs = []
        for epoch in range(10, 100):
            lr = sch.get_lr()
            lrs.append(lr)
            sch.step()
        lrs.append(sch.get_lr())

        self.assertEqual(lrs[0], 0.1)
        self.assertEqual(lrs[-1], 0.0)
        self.assertGreaterEqual(min(lrs[::]), 0.0)
        self.assertLessEqual(max(lrs[::]), 0.1)

    def test_get_exclude_from_weight_decay_fn(self):
        model = nn.Linear(10, 100, bias_attr=True)
        exclude_list = ['bias']
        fn = get_exclude_from_weight_decay_fn(exclude_list)
        # should return false if name in exclude_list 
        for name, param in model.named_parameters():
            if name.endswith('weight'):
                self.assertTrue(fn(name))
            elif name.endswith('bias'):
                self.assertFalse(fn(name))
