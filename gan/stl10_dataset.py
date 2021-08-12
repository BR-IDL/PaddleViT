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
STL-10 Dataset and related methods
"""
import os
import numpy as np
from PIL import Image
from paddle.io import Dataset


class STL10Dataset(Dataset):
    """paddle dataset for loading STL-10 binary data
    This class will load the binary file from STL-10 dataset,
    extract and read images and labels. Images are stored in numpy array,
    with shape: [num_images, 96,96,3]. Labels are store in numpy array, with
    shape: [num_images].

    Args:
        file_folder: str, folder path of STL-10 dataset binary files
        mode: str, dataset mode, choose from ['train', 'test'], default: 'train'
        transform: paddle.vision.transforms, transforms which is applied on data, default: None
    """
    def __init__(self, file_folder, mode='train', transform=None):
        super().__init__()
        assert mode in ['train', 'test']
        self.folder = file_folder
        self.transform = transform
        self.height = 96
        self.width = 96
        self.channels = 3
        # num of bytes of a single image
        self.image_bytes = self.height * self.width * self.channels
        self.train_filepath = os.path.join(file_folder, f'{mode}_X.bin')
        self.label_filepath = os.path.join(file_folder, f'{mode}_y.bin')

        self.images = read_all_images(self.train_filepath)
        self.labels = read_labels(self.label_filepath)
        print(f'----- STL-10 dataset {mode} len = {self.labels.shape[0]}')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        data = self.images[index]
        if self.transform is not None:
            data = self.transform(data)
        label = self.labels[index]
        return data, label


def read_labels(label_path):
    """read data labels from binary file
    Args:
        label_path: label binary file path, e.g.,'train_y.bin'
    Returns:
        labels: np.array, the label array with shape [num_images]
    """
    with open(label_path, 'rb') as infile:
        labels = np.fromfile(infile, dtype=np.uint8)
    return labels


def read_all_images(data_path):
    """read all images from binary file 
    Args:
        data_path: data binary file path, e.g.,'train_X.bin'
    Returns:
        images: np.array, the image array with shape [num_images, 96, 96, 3]
    """
    with open(data_path, 'rb') as infile:
        # read whole data in unit8
        data = np.fromfile(infile, dtype=np.uint8)
        # images are stored in column major order
        # 1st, 2nd, 3rd 96x96 are red, green, blue channels
        images = np.reshape(data, (-1, 3, 96, 96))
        # outputs are with shape [num_images, height, width, channels]
        images = np.transpose(images, (0, 3, 2, 1))
        return images
        

def save_image(image, name):
    im = Image.fromarray(image)
    im.save(f"{name}.png")


def save_images(images, labels, out_path):
    for idx, image in enumerate(images):
        out_path = os.path.join(out_path, str(labels[i]))
        os.makedirs(out_path, exist_ok=True)
        save_image(image, os.path.join(out_path, str(idx)+'.png'))


# NOTE: this is for test, can be removed later
if __name__ == "__main__":
    dataset = STL10Dataset(file_folder='./stl10_binary')
    print(dataset.labels.shape)
    for idx, (data, label) in enumerate(dataset):
        print(idx)
        print(data.shape)
        # save images to file
        save_image(data, f'{idx}.png')
        print(label)
        print('-----')
        if idx == 10:
            break
