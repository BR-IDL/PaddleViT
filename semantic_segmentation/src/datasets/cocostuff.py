#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

import os
import glob
from src.datasets import Dataset
from src.transforms import Compose


class CocoStuff(Dataset):
    """CocoStuff

    COCO-Stuff dataset `https://github.com/nightrome/cocostuff`.
    There are 164k images in COCO-stuff dataset that span over 172 categories 
    including 80 things, 91 stuff, and 1 unlabeled class.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str): Which part of dataset to use, train or val. Default: 'train'.
        num_classes (int): the number of classes
    """

    def __init__(self, transforms, dataset_root, mode='train', num_classes=172):
        super(CocoStuff, self).__init__(transforms=transforms, 
            num_classes=num_classes, dataset_root=dataset_root, mode=mode)
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = 255
        if mode not in ['train', 'val']:
            raise ValueError("mode should be 'train', 'val',"
                             "but got {}.".format(mode))
        img_dir = os.path.join(self.dataset_root, 'images')
        label_dir = os.path.join(self.dataset_root, 'annotations')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError("The dataset is not Found or the folder structure"
                             "is nonconfoumance.")
        label_files = sorted(
            glob.glob(os.path.join(label_dir, mode + '2017', '*.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode + '2017', '*.jpg')))
        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]
