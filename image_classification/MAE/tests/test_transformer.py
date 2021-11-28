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
import paddle.nn.functional as F
from config import *
from transformer import build_mae_pretrain
from transformer import PatchEmbedding
from transformer import Attention
from transformer import Mlp
from transformer import Encoder


class TransformerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.config = get_config()
        cls.dummy_img = np.random.randn(4, 3, 224, 224).astype('float32')
        cls.dummy_tensor = paddle.to_tensor(cls.dummy_img)
        cls.mae = build_mae_pretrain(cls.config)
        cls.mae.train()

    @classmethod
    def tearDown(cls):
        pass

    # @unittest.skip('skip for debug')
    def test_out_shape(self):
        reconstruct, mask = TransformerTest.mae(TransformerTest.dummy_tensor)
        self.assertEqual(reconstruct.shape, [4, 49, 768])
        self.assertEqual(mask.shape, [4, 49, 768])

    @unittest.skip('skip for debug')
    def test_all_parameters_updated(self):
        optim = paddle.optimizer.SGD(parameters=TransformerTest.mae.parameters(), learning_rate=0.1)
        reconstruct, masked_image = TransformerTest.mae(TransformerTest.dummy_tensor)
        loss = F.mse_loss(reconstruct, masked_image)
        loss.backward()

        for name, param in TransformerTest.mae.named_parameters():
            if not param.stop_gradient:
                self.assertIsNotNone(param.gradient())
                # self.assertNotEqual(0, np.sum(param.gradient() ** 2))

    # @unittest.skip('skip for debug')
    def test_embeddings(self):
        embed = PatchEmbedding()
        dummy_img = np.random.randn(4, 3, 224, 224).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)

        patch_out = embed.patch_embedding(dummy_tensor)
        embed_out = embed(dummy_tensor)
        self.assertEqual(patch_out.shape, [4, 768, 14, 14])
        self.assertEqual(embed.cls_token.shape, [1, 1, 768])
        self.assertEqual(embed_out.shape, [4, 14 * 14 + 1, 768])

    # @unittest.skip('skip for debug')
    def test_attention(self):
        attn_op = Attention(
            TransformerTest.config.MODEL.TRANS.ENCODER.EMBED_DIM,
            TransformerTest.config.MODEL.TRANS.ENCODER.NUM_HEADS,
            TransformerTest.config.MODEL.TRANS.QKV_BIAS)
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)

        out, attn = attn_op(dummy_tensor)
        self.assertEqual(attn.shape, [4, 12, 50, 50])
        self.assertEqual(out.shape, [4, 50, 768])

    def test_mlp(self):
        mlp_op = Mlp(
            TransformerTest.config.MODEL.TRANS.ENCODER.EMBED_DIM,
            TransformerTest.config.MODEL.TRANS.MLP_RATIO)
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)

        out = mlp_op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])

    def test_position_embedding_not_update(self):
        origin = TransformerTest.mae.position_embedding.get_encoder_embedding().clone()
        optim = paddle.optimizer.SGD(parameters=TransformerTest.mae.parameters(), learning_rate=0.1)
        reconstruct, masked_image = TransformerTest.mae(TransformerTest.dummy_tensor)
        loss = F.mse_loss(reconstruct, masked_image)
        loss.backward()
        optim.step()
        update = TransformerTest.mae.position_embedding.get_encoder_embedding().clone()
        self.assertTrue((origin.numpy() == update.numpy()).all())

    def test_encoder(self):
        encoder_op = Encoder(
            TransformerTest.config.MODEL.TRANS.ENCODER.EMBED_DIM,
            TransformerTest.config.MODEL.TRANS.ENCODER.NUM_HEADS,
            TransformerTest.config.MODEL.TRANS.ENCODER.DEPTH,
        )
        dummy_img = np.random.randn(4, 50, 768).astype('float32')
        dummy_tensor = paddle.to_tensor(dummy_img)

        out, _ = encoder_op(dummy_tensor)
        self.assertEqual(out.shape, [4, 50, 768])
