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
import argparse
from image_classification.CrossViT.config import update_config, get_config

class ConfigTest(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser('')
        parser.add_argument('-cfg', type=str, default=None)
        parser.add_argument('-dataset', type=str, default="cifar10")
        parser.add_argument('-batch_size', type=int, default=128)
        parser.add_argument('-image_size', type=int, default=256)
        parser.add_argument('-ngpus', type=int, default=None)
        parser.add_argument('-data_path', type=str, default='/cifar10/')
        parser.add_argument('-eval', action='store_false') # enable eval
        parser.add_argument('-pretrained', type=str, default='pretrained')
        parser.add_argument('-resume', type=str, default=None)
        parser.add_argument('-last_epoch', type=int, default=None)
        self.args = parser.parse_args()

    def tearDown(self):
        pass

    def test_update_config(self):
        config = get_config()
        config = update_config(config, self.args)

        self.assertEqual(config.DATA.DATASET, 'cifar10')
        self.assertEqual(config.DATA.BATCH_SIZE, 128)
        self.assertEqual(config.DATA.IMAGE_SIZE, 256)
        self.assertEqual(config.DATA.DATA_PATH, '/cifar10/')
        self.assertEqual(config.EVAL, True)
        self.assertEqual(config.DATA.BATCH_SIZE_EVAL, 128)
        self.assertEqual(config.MODEL.PRETRAINED, 'pretrained')

    def test_update_config_from_file(self):
        config = get_config()
        self.args.cfg = './tests/test_config.yaml'
        self.args.image_size = None
        self.args.ngpus = None
        config = update_config(config, self.args)

        self.assertEqual(config.DATA.IMAGE_SIZE, 384)
        self.assertEqual(config.DATA.CROP_PCT, 1.0)

        self.assertEqual(config.MODEL.TRANS.PATCH_SIZE, 16)
        self.assertEqual(config.MODEL.TRANS.EMBED_DIM, 768)
        self.assertEqual(config.MODEL.TRANS.MLP_RATIO, 4.0)
        self.assertEqual(config.MODEL.TRANS.DEPTH, 12)
        self.assertEqual(config.MODEL.TRANS.NUM_HEADS, 12)
        self.assertEqual(config.MODEL.TRANS.QKV_BIAS, True)

        self.assertEqual(config.MODEL.NAME, 'crossvit_base_224')
        self.assertEqual(config.MODEL.TYPE, 'ViT')

    def test_get_config(self):
        config1 = get_config()
        config2 = get_config()
        self.assertEqual(config1, config2)
