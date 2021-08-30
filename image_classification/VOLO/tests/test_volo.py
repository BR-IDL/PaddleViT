import unittest
import numpy as np
import paddle
import paddle.nn as nn
from config import *
from volo import *


class VoloTest(unittest.TestCase):
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
        out = layer(VoloTest.dummy_tensor)
        self.assertTrue(np.allclose(out.numpy(), VoloTest.dummy_tensor.numpy()))

    #@unittest.skip('skip for debug')
    def test_downsample(self):
        layer = Downsample(3, 16, 4)
        tensor = paddle.randn(shape=[4, 256, 256, 3])
        out = layer(tensor)
        self.assertEqual([4, 64, 64, 16], out.shape)

    def test_patchembedding(self):
        layer = PatchEmbedding(stem_conv=True)
        tensor = paddle.randn(shape=[4, 3, 224, 224])
        out = layer(tensor)
        self.assertEqual([4, 384, 28, 28], out.shape)

    def test_mlp(self):
        layer = Mlp(in_features=128, hidden_features=64, dropout=0.1)
        tensor = paddle.randn(shape=[4, 128])
        out = layer(tensor)
        self.assertEqual([4, 128], out.shape)

    def test_outlooker_attention(self):
        layer = OutlookerAttention(dim=64,  num_heads=8)
        tensor = paddle.randn(shape=[4, 32, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 32, 32, 64], out.shape)

    def test_outlooker(self):
        layer = Outlooker(dim=64, kernel_size=3, padding=1, num_heads=8)
        tensor = paddle.randn(shape=[4, 32, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 32, 32, 64], out.shape)

    def test_attention(self):
        layer = Attention(dim=64, num_heads=8)
        tensor = paddle.randn(shape=[4, 32, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 32, 32, 64], out.shape)

    def test_transformer(self):
        layer = Transformer(dim=64, num_heads=8)
        tensor = paddle.randn(shape=[4, 32, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 32, 32, 64], out.shape)

    def test_class_attention(self):
        layer = ClassAttention(dim=64)
        tensor = paddle.randn(shape=[4, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 1, 64], out.shape)

    def test_class_block(self):
        layer = ClassBlock(dim=64, num_heads=8)
        tensor = paddle.randn(shape=[4, 32, 64])
        out = layer(tensor)
        self.assertEqual([4, 32, 64], out.shape)

    @unittest.skip('skip for debug')
    def test_build_model(self):
        print(VoloTest.config)
        model = build_volo(VoloTest.config)
        print(model)

    @unittest.skip('skip for debug')
    def test_model_inference(self):
        print(VoloTest.config)
        model = build_volo(VoloTest.config)
        print(model(VoloTest.dummy_tensor))

