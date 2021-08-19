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

"""
CelebA Dataset related classes and methods
Currently only support for GAN
"""

import os
import glob
from PIL import Image
from paddle.io import Dataset

class CelebADataset(Dataset):
    """Build CelebA dataset

    This class gets train/val imagenet datasets, which loads transfomed data and labels.

    Attributes:
        file_folder: path where align and cropped images are stored
        transform: preprocessing ops to apply on image
    """

    def __init__(self, file_folder, transform=None):
        """CelebA Dataset with dataset file path, and transform"""
        super().__init__()
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = glob.glob(os.path.join(file_folder, '*.jpg'))
        print(f'----- CelebA img_align len = {len(self.img_path_list)}')

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

#if __name__ == "__main__":
#    dataset = CelebADataset(file_folder='./celeba/img_align_celeba')
#    for idx, (data, label) in enumerate(dataset):
#        print(idx)
#        print(data.size)
#        print('-----')
#        if idx == 10:
#            break
