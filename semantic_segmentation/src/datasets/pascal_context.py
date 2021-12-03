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
from PIL import Image
from src.datasets import Dataset
from src.transforms import Compose


class PascalContext(Dataset):
    """PascalContext

    This dataset is a set of additional annotations for PASCAL VOC 2010. It goes
    beyond the original PASCAL semantic segmentation task by providing annotations 
    for the whole scene. The statistics section has a full list of 400+ labels. 
    Here, we choose 59 foreground and 1 background class for training segmentation 
    models. (The ``img`` is fixed to '.jpg' and ``label`` is fixed to '.png'.)

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str): Which part of dataset to use. ('train', 'trainval', 
                    'context', 'val').
        num_classes (int): the number of classes
    """

    def __init__(self, transforms=None, dataset_root=None, mode='train', 
            num_classes=60):
        super(PascalContext, self).__init__(transforms=transforms, 
            num_classes=num_classes, dataset_root=dataset_root, mode=mode)

        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = num_classes
        self.ignore_index = 255

        if mode not in ['train', 'trainval', 'val']:
            raise ValueError("`mode` should be one of ('train', 'trainval', 'val')"
                             "in PascalContext dataset, but got {}.".format(mode))

        if self.dataset_root is None:
            raise ValueError("the path of this dataset is None")
        
        image_set_dir = os.path.join(
            self.dataset_root, 'ImageSets', 'SegmentationContext')

        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val.txt')
            #file_path = os.path.join(image_set_dir, 'val_mini.txt')
        elif mode == 'trainval':
            file_path = os.path.join(image_set_dir, 'trainval.txt')
        print("file_path: ", file_path)
        if not os.path.exists(file_path):
            raise RuntimeError("PASCAL-Context annotations are not ready.")

        img_dir = os.path.join(self.dataset_root, 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'SegmentationClassContext')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])
        print("mode: {}, file_nums: {}".format(mode, len(self.file_list)))
