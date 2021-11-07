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
from ..utils import OneCycleLRScheduler


class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_one_cycle_lr(self):
        total_steps = 1000
        max_lr = 1e-5
        scheduler = OneCycleLRScheduler(learning_rate=1, max_lr=max_lr, total_steps=total_steps)
        lr_list = list()
        for i in range(total_steps):
            lr_list.append(scheduler.get_lr())
            scheduler.step()

        self.assertAlmostEqual(max_lr, lr_list[int(total_steps * 0.4)], places=8)
        self.assertAlmostEqual(max_lr / 20, lr_list[0], places=8)
        self.assertAlmostEqual(max_lr / 20, lr_list[int(total_steps * 0.8)], places=8)
        self.assertAlmostEqual(0, lr_list[-1], places=8)

    def test_one_cycle_lr_with_last_epoch(self):
        total_steps = 1000
        max_lr = 1e-5
        last_epoch = 399
        scheduler = OneCycleLRScheduler(learning_rate=1, max_lr=max_lr, total_steps=total_steps,
                                        last_epoch=last_epoch)
        lr_list = list()
        for i in range(total_steps - last_epoch):
            lr_list.append(scheduler.get_lr())
            scheduler.step()

        self.assertAlmostEqual(0, lr_list[-1], places=8)
        self.assertAlmostEqual(max_lr, lr_list[0], places=8)
        self.assertAlmostEqual(max_lr / 20, lr_list[int(total_steps * 0.4)], places=8)


if __name__ == '__main__':
    unittest.main()
