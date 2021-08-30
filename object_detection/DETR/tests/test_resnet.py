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
from resnet import resnet50 as myresnet50
from paddle.vision.models import resnet50
from resnet import resnet18 as myresnet18
from paddle.vision.models import resnet18

from backbone import FrozenBatchNorm2D
from backbone import IntermediateLayerGetter


class ResnetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')
        cls.dummy_img = np.random.randn(1, 3, 224, 224).astype('float32')
        cls.dummy_tensor = paddle.to_tensor(cls.dummy_img)

    @classmethod
    def tearDown(cls):
        pass


    @unittest.skip('skip for debug')
    def test_resnet50(self):
        model = resnet50(pretrained=True) 
        mymodel = myresnet50(pretrained=True)

        out = model(ResnetTest.dummy_tensor)
        myout = mymodel(ResnetTest.dummy_tensor)
    
        out = out.cpu().numpy()
        myout = myout.cpu().numpy()

        self.assertTrue(np.allclose(out, myout))

    @unittest.skip('skip for debug')
    def test_resnet18(self):
        model = resnet18(pretrained=True) 
        mymodel = myresnet18(pretrained=True)

        out = model(ResnetTest.dummy_tensor)
        myout = mymodel(ResnetTest.dummy_tensor)
    
        out = out.cpu().numpy()
        myout = myout.cpu().numpy()


        self.assertTrue(np.allclose(out, myout))




    @unittest.skip('skip for debug')
    def test_frozen_bn(self):
        model = resnet18(pretrained=True) 
        bn1 = model.bn1
        bn1_st = bn1.state_dict()
        bn1.eval()

        frozen_bn = FrozenBatchNorm2D(64)
        frozen_bn.set_state_dict(bn1_st)

        tmp = paddle.randn([4, 64, 5, 5])
        out = bn1(tmp)
        out_f = frozen_bn(tmp)

        self.assertTrue([4, 64, 5, 5], out_f.shape)

        out = out.cpu().numpy()
        out_f = out_f.cpu().numpy()

        self.assertTrue(np.allclose(out, out_f, atol=1e-5))




    @unittest.skip('skip for debug')
    def test_intermediate_layer_getter(self):
        model = resnet50(pretrained=True)
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        new_model = IntermediateLayerGetter(model, return_layers)
        tmp = paddle.randn([1, 3, 224, 224])
        out = new_model(tmp)
        #print([(k, v.shape) for k,v in out.items()])
    
        self.assertEqual(out['0'].shape, [1, 256, 56, 56])
        self.assertEqual(out['1'].shape, [1, 512, 28, 28])
        self.assertEqual(out['2'].shape, [1, 1024, 14, 14])
        self.assertEqual(out['3'].shape, [1, 2048, 7, 7])





