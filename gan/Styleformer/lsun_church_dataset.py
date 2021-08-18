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
LSUN-church Dataset and related methods
"""
import os
import io
import numpy as np
import lmdb
from PIL import Image
from paddle.io import Dataset


class LSUNchurchDataset(Dataset):
    """paddle dataset for loading LSUN-church binary data
    This class will load the lmdb file from LSUN-church dataset,
    extract and read images. Images are stored in list of numpy array

    Args:
        file_folder: str, folder path of LSUN-church dataset lmdb
        mode: str, dataset mode, choose from ['train', 'val'], default: 'train'
        transform: paddle.vision.transforms, transforms which is applied on data, default: None
        max_num_images: int, num of images used in the dataset,
        if None, use all the images, default: None
    """
    def __init__(self, file_folder, mode='train', transform=None, max_num_images=None):
        super().__init__()
        assert mode in ['train', 'val']
        self.transform = transform
        self.file_folder = file_folder
        with lmdb.open(file_folder,
                       map_size=1099511627776,
                       max_readers=32,
                       readonly=True,
                       readahead=False,
                       meminit=False,
                       lock=False).begin(write=False) as txn:
            self.num_images = txn.stat()['entries']
            # efficient way of loading keys only
            self.keys = list(txn.cursor().iternext(values=False))

        if max_num_images is not None:
            self.num_images = min(self.num_images, max_num_images)

        print(f'----- LSUN-church dataset {mode} len = {self.num_images}')

    def open_lmdb(self):
        """ Open lmdb, this method is called in __getitem__ method
        Note that lmdb is not opened in __init__ method, to support multi-process.
        Reference: https://github.com/pytorch/vision/issues/689
        """
        self.env = lmdb.open(self.file_folder,
                             max_readers=32,
                             readonly=True,
                             readahead=False,
                             meminit=False,
                             lock=False)
        self.txn = self.env.begin(buffers=True)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        key = self.keys[index]
        image_bytes = self.txn.get(key)
        image = read_image(image_bytes)
        if self.transform is not None:
            image = self.transform(image)
        label = 0
        return image, label


def read_image(image_bytes):
    """read image from bytes loaded from lmdb file
    Args:
        image_bytes: bytes, image data in bytes
    Returns:
        image: np.array, stores the image with shape [h, w, c]
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    return image


def save_image(image, name):
    img = Image.fromarray(image)
    img.save(f"{name}.png")


def save_images(images, labels, out_path):
    for idx, image in enumerate(images):
        out_path = os.path.join(out_path, str(labels[idx]))
        os.makedirs(out_path, exist_ok=True)
        save_image(image, os.path.join(out_path, str(idx)))


## NOTE: this is for test, can be removed later
#if __name__ == "__main__":
#    dataset = LSUNchurchDataset(file_folder='./church_outdoor_train_lmdb')
#    for idx, (data, label) in enumerate(dataset):
#        print(idx)
#        print(data.shape)
#        # save images to file
#        save_image(data, f'lsun_{idx}')
#        print('-----')
#        if idx == 10:
#            break
