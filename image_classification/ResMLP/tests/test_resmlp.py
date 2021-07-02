# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
from config import *
from resmlp import Identity
from resmlp import PatchEmbedding
from resmlp import ResMlp
from resmlp import Mlp
from resmlp import ResBlock
from resmlp import Affine
from resmlp import build_res_mlp


class MlpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.config = get_config()
        cls.dummy_img = np.random.randn(4, 3, 224, 224).astype('float32')
        cls.dummy_tensor = paddle.to_tensor(cls.dummy_img)
        cls.model = build_res_mlp(cls.config)

    @classmethod
    def tearDown(cls):
        pass
    
    #@unittest.skip('skip for debug')
    def test_out_shape(self):
        out = MlpTest.model(MlpTest.dummy_tensor)
        self.assertEqual(out.shape, [4, 1000])

    #@unittest.skip('skip for debug')
    def test_all_parameters_updated(self):
        optim = paddle.optimizer.SGD(
            parameters=MlpTest.model.parameters(), learning_rate=0.1)
        out = MlpTest.model(MlpTest.dummy_tensor)
        loss = out.mean()
        loss.backward()
        optim.step()
    
        for name, param in MlpTest.model.named_parameters():
            if not param.stop_gradient:
                self.assertIsNotNone(param.gradient())
                self.assertNotEqual(0, np.sum(param.gradient()**2))
    
    #@unittest.skip('skip for debug')
    def test_embeddings(self):
        embed = PatchEmbedding(embed_dim=768)
        dummy_img = np.random.randn(4, 3, 224, 224).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)
    
        embed_out = embed(dummy_tensor)
        self.assertEqual(embed_out.shape, [4, 3136, 768])

    #@unittest.skip('skip for debug')
    def test_mlp(self):
        mlp_op = Mlp(768, 256, 0.0)
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)
    
        out = mlp_op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])

    #@unittest.skip('skip for debug')
    def test_identity(self):
        op = Identity()
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)
    
        out = op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])
    
    #@unittest.skip('skip for debug')
    def test_mixer_block(self):
        op = ResBlock(dim=768, seq_len=50)
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)
    
        out = op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])

    def test_affine(self):
        op = Affine(dim=768)
        dummy_tensor = paddle.ones([4, 50, 768])

        dummy_alpha = paddle.ones([1, 1, 768]) * 0.5 
        dummy_beta = paddle.ones([1, 1, 768]) * 0.2 

        op.alpha.set_value(dummy_alpha)
        op.beta.set_value(dummy_beta)
    
        out = op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])
        self.assertEqual(paddle.equal_all(out, paddle.ones([4, 50, 768]) * 0.7).numpy(), True)
