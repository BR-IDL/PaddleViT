import unittest
import numpy as np
import paddle
import paddle.nn as nn
from config import *
from cait import *


class CaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.config = get_config()
        cls.dummy_img = np.random.randn(4, 3, 224, 224).astype('float32')
        cls.dummy_tensor = paddle.to_tensor(cls.dummy_img)
    
    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_identity(self):
        layer = Identity()
        out = layer(CaitTest.dummy_tensor)
        self.assertTrue(np.allclose(out.numpy(), CaitTest.dummy_tensor.numpy()))

    def test_patchembedding(self):
        layer = PatchEmbedding()
        tensor = paddle.randn(shape=[4, 3, 224, 224])
        out = layer(tensor)
        self.assertEqual([4, 3136, 96], out.shape)

    def test_mlp(self):
        layer = Mlp(in_features=128, hidden_features=64, dropout=0.1)
        tensor = paddle.randn(shape=[4, 128])
        out = layer(tensor)
        self.assertEqual([4, 128], out.shape)

    def test_talkinghead_attention(self):
        layer = TalkingHeadAttention(dim=64,  num_heads=8)
        tensor = paddle.randn(shape=[4, 196, 64])
        out = layer(tensor)
        self.assertEqual([4, 196, 64], out.shape)

    def test_layer_scale_block(self):
        layer = LayerScaleBlock(dim=64, num_heads=8)
        tensor = paddle.randn(shape=[4, 196, 64])
        out = layer(tensor)
        self.assertEqual([4, 196, 64], out.shape)

    def test_class_attention(self):
        layer = ClassAttention(dim=64)
        tensor = paddle.randn(shape=[4, 196, 64])
        out = layer(tensor)
        self.assertEqual([4, 1, 64], out.shape)

    def test_layer_scale_block_class_attention(self):
        layer = LayerScaleBlockClassAttention(dim=64, num_heads=8)
        tensor = paddle.randn(shape=[4, 196, 64])
        cls_tensor = paddle.randn(shape=[4, 1, 64])
        out = layer(tensor, cls_tensor)
        self.assertEqual([4, 1, 64], out.shape)

    #@unittest.skip('skip for debug')
    def test_build_model(self):
        print(CaitTest.config)
        model = build_cait(CaitTest.config)
        print(model)

    #@unittest.skip('skip for debug')
    def test_model_inference(self):
        print(CaitTest.config)
        model = build_cait(CaitTest.config)
        print(model(CaitTest.dummy_tensor))

