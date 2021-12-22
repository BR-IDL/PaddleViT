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

"""
random mask generator for MAE pretraining
"""

import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, with_cls_token=True):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2

        self.height  = input_size[0]
        self.width = input_size[1]
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.with_cls_token = with_cls_token

    def __call__(self):
        mask = np.hstack([np.zeros(self.num_patches - self.num_mask),
                          np.ones(self.num_mask)])
        np.random.shuffle(mask)
        if self.with_cls_token:
            mask = np.insert(mask, 0, 0)
        return mask


#def main():
#    rmg = RandomMaskingGenerator(input_size=32, mask_ratio=0.75)
#    mask = rmg()
#    for v in mask:
#        print(v, end=', ')
#
#if __name__ == "__main__":
#    main()
