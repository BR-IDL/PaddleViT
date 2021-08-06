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
    """
    def __init__(self, file_folder, mode='train', transform=None):
        super().__init__()
        assert mode in ['train', 'val']
        self.transform = transform
        self.images = read_all_images(file_folder)
        print(f'----- LSUN-church dataset {mode} len = len{self.images}')

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        data = self.images[index]
        if self.transform is not None:
            data = self.transform(data)
        label = 0
        return data, label


def read_all_images(data_path):
    """read all images from lmdb file 
    Args:
        data_path: data lmdb folder path, e.g.,'church_outdoot_train_lmdb'
    Returns:
        images: list of np.array, list stores all the images with shape [h, w, c] 
    """
    env = lmdb.open(data_path, map_size=1099511627776, max_readers=100, readonly=True)
    images = []
    with env.begin(write=False) as context:
        cursor = context.cursor()
        for key, val in cursor:
            img = Image.open(io.BytesIO(val))
            img = np.array(img)
            images.append(img)
    return images
        

def save_image(image, name):
    im = Image.fromarray(image)
    im.save(f"{name}.png")


def save_images(images, labels, out_path):
    for idx, image in enumerate(images):
        out_path = os.path.join(out_path, str(labels[i]))
        os.makedirs(out_path, exist_ok=True)
        save_image(image, os.path.join(out_path, str(idx)))


# NOTE: this is for test, can be removed later
if __name__ == "__main__":
    dataset = LSUNchurchDataset(file_folder='./church_outdoor_val_lmdb')
    for idx, (data, label) in enumerate(dataset):
        print(idx)
        print(data.shape)
        # save images to file
        save_image(data, f'lsun_{idx}')
        print('-----')
        if idx == 10:
            break
