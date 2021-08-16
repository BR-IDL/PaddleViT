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
from config import *
from datasets import *
from paddle.io import DataLoader
#from multiprocessing import SimpleQueue

#paddle.set_device('cpu')

class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser('')
        parser.add_argument('-cfg', type=str, default=None)
        parser.add_argument('-dataset', type=str, default='imagenet2012')
        parser.add_argument('-batch_size', type=int, default=4)
        parser.add_argument('-image_size', type=int, default=224)
        parser.add_argument('-ngpus', type=int, default=None)
        parser.add_argument('-data_path', type=str, default='/dataset/imagenet')
        parser.add_argument('-eval', action='store_true')
        parser.add_argument('-pretrained', type=str, default=None)
        parser.add_argument('-resume', type=str, default=None)
        parser.add_argument('-last_epoch', type=int, default=None)
        cls.args = parser.parse_args()
        cls.config = get_config()
        cls.config = update_config(cls.config, cls.args)

        cls.dataset_train = get_dataset(DatasetTest.config, mode='train')
        cls.dataset_test = get_dataset(DatasetTest.config, mode='val')

    @classmethod 
    def tearDown(cls):
        pass

    @unittest.skip('skip for debug')
    def test_shape(self):
        sample = next(iter(DatasetTest.dataset_train))
        self.assertEqual([3, 224, 224], sample[0].shape)

        sample = next(iter(DatasetTest.dataset_test))
        self.assertEqual([3, 224, 224], sample[0].shape)
    
    @unittest.skip('skip for debug')
    def test_scaling(self):
        sample = next(iter(DatasetTest.dataset_train))[0]
        self.assertTrue(paddle.any(sample < 0))
        self.assertTrue(paddle.any(sample > 0))
        self.assertGreaterEqual(1, sample.max().cpu().numpy())
        self.assertLessEqual(-1, sample.min().cpu().numpy())

        sample = next(iter(DatasetTest.dataset_test))[0]
        self.assertGreaterEqual(1, sample.max().cpu().numpy())
        self.assertLessEqual(-1, sample.min().cpu().numpy())
        self.assertTrue(paddle.any(sample < 0))
        self.assertTrue(paddle.any(sample > 0))

    @unittest.skip('skip for debug')
    def test_single_process_dataloader(self):
        self._test_loader(DatasetTest.dataset_train, 'train', False)
        self._test_loader(DatasetTest.dataset_test, 'test', False)

    def _test_loader(self, dataset, mode, multi_process):
        dataloader = get_dataloader(DatasetTest.config,
                                    dataset,
                                    mode=mode,
                                    multi_process=multi_process)
        for idx, _ in enumerate(dataloader):
            if idx > 0 and idx % 1 == 0:
                print(f'----- test single process dataloader: {idx}/{len(dataloader)}')
            if idx == 10:
                return

    @unittest.skip('skip for debug')
    def test_multi_process_dataloader(self):
        tester = Tester()
        tester.run()
        self.assertEqual(tester.n_samples, 50000) 
            



class Tester:
    def __init__(self):
        parser = argparse.ArgumentParser('')
        parser.add_argument('-cfg', type=str, default=None)
        parser.add_argument('-dataset', type=str, default='imagenet2012')
        parser.add_argument('-batch_size', type=int, default=256)
        parser.add_argument('-image_size', type=int, default=224)
        parser.add_argument('-data_path', type=str, default='/dataset/imagenet/')
        parser.add_argument('-eval', action='store_false') # set test batch size
        parser.add_argument('-pretrained', type=str, default=None)
        args = parser.parse_args()
        self.config = get_config()
        self.config = update_config(self.config, args)
        self.dataset_train = get_dataset(self.config, mode='train')
        self.dataset_test = get_dataset(self.config, mode='val')
        self.n_samples = 0

    def run(self, mode='test'):
        # https://github.com/PaddlePaddle/Paddle/blob/5d8e4395b61929627151f6fd4a607589288a78bf/python/paddle/distributed/spawn.py#L272
        context = dist.spawn(self.main_worker, args=(mode,))
        self.n_samples = context.return_queues[0].get()
        print(f'----- total samples: {self.n_samples}')

    def main_worker(self, *args):
        mode = args[0]
        dist.init_parallel_env()
        local_rank = dist.get_rank()
        if mode == 'train':
            n_samples = self._test_loader(self.config, self.dataset_train, 'train', True) 
        else:
            n_samples = self._test_loader(self.config, self.dataset_test, 'test', True) 

        n_samples = paddle.to_tensor(np.array([n_samples]))
        dist.reduce(n_samples, 0)
        if local_rank == 0:
            return n_samples.cpu().numpy()


    def _test_loader(self, config, dataset, mode, multi_process):
        n_samples = 0
        dataloader = get_dataloader(config,
                                    dataset,
                                    mode=mode,
                                    multi_process=multi_process)
        local_rank = dist.get_rank()
        for idx, data in enumerate(dataloader):
            if idx > 0 and idx % 1 == 0:
                print(f'----- test single process({local_rank}) dataloader: {idx}/{len(dataloader)}')
                #print(local_rank, data[1])
            n_samples += data[0].shape[0] 

        return n_samples
