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
# limitations under the License.:



import unittest
import paddle
import numpy as np
from transformer import Transformer
from position_embedding import build_position_encoding
from utils import NestedTensor


class TransformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.tensors = paddle.randn((4, 256, 24, 33))
        cls.masks = paddle.ones((4, 24, 33))
        cls.query_embed = paddle.randn((100, 256))
        cls.pos_embed = paddle.randn((4, 256, 24, 33))

    @classmethod
    def tearDown(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass



    @unittest.skip('skip fo debug')
    def test_position_embed(self):
        t = TransformerTest.tensors
        m = TransformerTest.masks
        tensor_list = NestedTensor(t, m)

        pos_embed = build_position_encoding()
        out = pos_embed(tensor_list)
        self.assertEqual(out.shape, [4, 256, 24, 33])


    @unittest.skip('skip fo debug')
    def test_transformer(self):
        t = TransformerTest.tensors
        m = TransformerTest.masks
        q = TransformerTest.query_embed
        p = TransformerTest.pos_embed

        model = Transformer()
        out = model(src=t,
                    mask=m,
                    query_embed=q,
                    pos_embed=p)

    @unittest.skip('skip fo debug')
    def test_position_embed_sine(self):
        pass
    


