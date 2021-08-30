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
from config import *
from datasets import *


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')

    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_dataset(self):
        config = get_config()
        transforms = get_train_transforms(config)
        dataset = ImageNet2012Dataset(file_folder='/dataset/imagenet/', mode='train', transform=transforms)
        for idx, (data, label) in enumerate(dataset):
            self.assertEqual([3, 224, 224], data.shape)
            if idx == 10:
                return



