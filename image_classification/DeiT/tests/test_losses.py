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
from losses import SoftTargetCrossEntropyLoss
from losses import LabelSmoothingCrossEntropyLoss
from losses import DistillationLoss


class DummyModel(nn.Layer):
    def __init__(self, kd=False):
        super().__init__()
        self.layer = nn.Linear(8, 16)
        self.head = nn.Linear(16, 1000)
        self.head_kd = nn.Linear(16, 1000)
        self.kd = kd

    def forward(self, x):
        feature = self.layer(x)
        out = self.head(feature)
        if self.kd:
            out_kd = self.head_kd(feature)
            return out, out_kd
        else:
            return out


class LossesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.num_classes = 1000
        cls.batch_size = 4
        cls.x = paddle.rand(shape=[cls.batch_size, cls.num_classes])
        cls.target = cls.x.argmax(axis=-1)

    @classmethod
    def tearDown(cls):
        pass

    @unittest.skip('skip for debug')
    def test_soft_target_crossentropy_loss(self):
        x = LossesTest.x
        target = LossesTest.target
        soft_target = paddle.zeros([LossesTest.batch_size, LossesTest.num_classes])
        rand_idx = np.random.randint(0, LossesTest.num_classes - 1, size=LossesTest.batch_size) 
        for i, idx in enumerate(rand_idx):
            soft_target[i, idx] = 0.6
            soft_target[i, idx + 1] = 0.4

        criterion = SoftTargetCrossEntropyLoss()
        loss = criterion(x, soft_target)

        logprob = -nn.functional.log_softmax(x, axis=-1)
        true_loss = []
        for i, idx in enumerate(rand_idx):
            true_loss.append(logprob[i,int(idx)] * 0.6 + logprob[i, int(idx+1)] * 0.4)
        true_loss = np.array(true_loss).mean()

        self.assertAlmostEqual(loss.numpy()[0], true_loss, delta=1e-5)

    
    #@unittest.skip('skip for debug')
    def test_label_smoothing_crossentropy_loss(self):
        x = paddle.to_tensor([[0.2, 0.3, 0.4, 0.1],[0.6, 0.2, 0.1, 0.1]])
        target = paddle.to_tensor([2, 1])
        criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.3)
        loss = criterion(x, target)

        val = -paddle.nn.functional.log_softmax(x, axis=-1)
        true_loss = val[0][2] * 0.7  + val[1][1] * 0.7 + val[0,:].mean() * 0.3 + val[1,:].mean()* 0.3 
        true_loss = true_loss/2.0

        self.assertAlmostEqual(true_loss.numpy()[0], loss.numpy()[0], delta=1e-5)


    #@unittest.skip('skip for debug')
    def test_distillation_loss(self):
        model = DummyModel(kd=True)
        teacher_model = DummyModel(kd=False)
        x = paddle.randn([4, 8])
        out, out_kd = model(x)
        labels = paddle.randint(0, 999, [4])

        base_criterion = nn.CrossEntropyLoss()
        criterion = DistillationLoss(base_criterion,
                                     teacher_model,
                                     'none',
                                     alpha=0.3,
                                     tau=0.8)
        loss = criterion(x, (out, out_kd), labels)
        self.assertEqual(loss.shape, [1])

        criterion = DistillationLoss(base_criterion,
                                     teacher_model,
                                     'hard',
                                     alpha=0.3,
                                     tau=0.8)
        loss = criterion(x, (out, out_kd), labels)
        self.assertEqual(loss.shape, [1])

        criterion = DistillationLoss(base_criterion,
                                     teacher_model,
                                     'soft',
                                     alpha=0.3,
                                     tau=0.8)
        loss = criterion(x, (out, out_kd), labels)
        self.assertEqual(loss.shape, [1])


