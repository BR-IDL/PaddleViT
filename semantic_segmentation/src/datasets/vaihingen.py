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
import numpy as np
from PIL import Image
from src.datasets import Dataset
from src.transforms import Compose
import src.transforms.functional as F


class Vaihingen(Dataset):
    """Vaihingen 
    
    ISPRS 2D Semantic Labeling Contest
    https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/
    
    aistudio: https://aistudio.baidu.com/aistudio/datasetdetail/103733
   
    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The Vaihingen dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be 
        one of ('train', 'val'). Default: 'train'.
        num_classes (int): the number of classes
    """

    def __init__(self, transforms, dataset_root=None, mode='train', num_classes=6):
        super(Vaihingen, self).__init__(
            transforms=transforms, num_classes=num_classes, 
            dataset_root=dataset_root, mode=mode)
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = num_classes
        self.ignore_index = 255

        if mode not in ['train', 'val']:
            raise ValueError("`mode` should be one of ('train', 'val') in "
                             "Vaihingen dataset, but got {}.".format(mode))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if mode == 'train':
            img_dir = os.path.join(self.dataset_root, 'images/training')
            label_dir = os.path.join(self.dataset_root, 'annotations/training')
        elif mode == 'val':
            img_dir = os.path.join(self.dataset_root, 'images/validation')
            label_dir = os.path.join(self.dataset_root,'annotations/validation')
        img_files = os.listdir(img_dir)
        label_files = [i.replace('.tif', '_noBoundary.png') for i in img_files]
        for i in range(len(img_files)):
            img_path = os.path.join(img_dir, img_files[i])
            label_path = os.path.join(label_dir, label_files[i])
            self.file_list.append([img_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'val':
            img, _ = self.transforms(img=image_path)
            label = np.asarray(Image.open(label_path))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            label = label - 1
            label = label[np.newaxis, :, :]
            return img, label
        else:
            img, label = self.transforms(img=image_path, label=label_path)
            label = label - 1
            # Recover the ignore pixels adding by transform
            label[label == 254] = 255
            return img, label
