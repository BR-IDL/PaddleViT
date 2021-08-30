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
import copy
import numpy as np
import paddle
import paddle.nn as nn
from model_ema import ModelEma


class DummyModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 8)
        self.head = nn.Linear(8, 5)

    def forward(self, x):
        feature = self.layer(x)
        out = self.head(feature)
        return out


class ModelEmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')

    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_model_ema(self):
        model = DummyModel()
        criterion = nn.CrossEntropyLoss()
        optim = paddle.optimizer.SGD(learning_rate=0.1,
                                       parameters=model.parameters())
        model_ema = ModelEma(model, decay=0.5)
        prev_weight = copy.deepcopy(model_ema.module.head.weight)
        for i in range(5):
            x = paddle.rand([4, 4])
            target = paddle.randint(0, 5, [4])
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optim.step()
            optim.clear_grad()
            model_ema.update(model)
            # test model ema update
            model_ema_weight = model_ema.module.head.weight
            self.assertFalse(np.allclose(prev_weight.numpy(), model_ema_weight.numpy(), atol=1e-5))
            prev_weight = copy.deepcopy(model_ema.module.head.weight)

            
