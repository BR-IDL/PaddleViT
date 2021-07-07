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
Dataset related classes and methods for ViT training and validation
Cifar10, Cifar100 and ImageNet2012 are supported
"""

import os
import math
from paddle.io import Dataset, DataLoader, DistributedBatchSampler
from paddle.vision import transforms, datasets, image_load

class ImageNet2012Dataset(Dataset):
    """Build ImageNet2012 dataset

    This class gets train/val imagenet datasets, which loads transfomed data and labels.

    Attributes:
        file_folder: path where imagenet images are stored
        transform: preprocessing ops to apply on image
        img_path_list: list of full path of images in whole dataset
        label_list: list of labels of whole dataset
    """

    def __init__(self, file_folder, mode="train", transform=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super(ImageNet2012Dataset, self).__init__()
        assert mode in ["train", "val"]
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = []
        self.label_list = []

        if mode == "train":
            self.list_file = os.path.join(self.file_folder, "train_list.txt")
        else:
            self.list_file = os.path.join(self.file_folder, "val_list.txt")

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder, img_path))
                self.label_list.append(img_label)
        print(f'----- Imagenet2012 image {mode} list len = {len(self.label_list)}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        data = self.transform(data)
        label = self.label_list[index]

        return data, label


def get_train_transforms(config):
    """ Get training transforms

    For training, a RandomResizedCrop is applied, then normalization is applied with
    [0.5, 0.5, 0.5] mean and std. The input pixel values must be rescaled to [0, 1.]
    Outputs is converted to tensor

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: training transforms
    """

    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
                                     scale=(0.05, 1.0)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms_train


def get_val_transforms(config):
    """ Get training transforms

    For validation, image is first Resize then CenterCrop to image_size.
    Then normalization is applied with [0.5, 0.5, 0.5] mean and std.
    The input pixel values must be rescaled to [0, 1.]
    Outputs is converted to tensor

    Args:
        config: configs contains IMAGE_SIZE, see config.py for details
    Returns:
        transforms_train: training transforms
    """

    scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
    transforms_val = transforms.Compose([
        transforms.Resize(scale_size, interpolation='bicubic'),
        transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms_val


def get_dataset(config, mode='train'):
    """ Get dataset from config and mode (train/val)

    Returns the related dataset object according to configs and mode(train/val)

    Args:
        config: configs contains dataset related settings. see config.py for details
    Returns:
        dataset: dataset object
    """

    assert mode in ['train', 'val']
    if config.DATA.DATASET == "cifar10":
        if mode == 'train':
            dataset = datasets.Cifar10(mode=mode, transform=get_train_transforms(config))
        else:
            dataset = datasets.Cifar10(mode=mode, transform=get_val_transforms(config))
    elif config.DATA.DATASET == "cifar100":
        if mode == 'train':
            dataset = datasets.Cifar100(mode=mode, transform=get_train_transforms(config))
        else:
            dataset = datasets.Cifar100(mode=mode, transform=get_val_transforms(config))
    elif config.DATA.DATASET == "imagenet2012":
        if mode == 'train':
            dataset = ImageNet2012Dataset(config.DATA.DATA_PATH,
                                          mode=mode,
                                          transform=get_train_transforms(config))
        else:
            dataset = ImageNet2012Dataset(config.DATA.DATA_PATH,
                                          mode=mode,
                                          transform=get_val_transforms(config))
    else:
        raise NotImplementedError(
            "[{config.DATA.DATASET}] Only cifar10, cifar100, imagenet2012 are supported now")
    return dataset


def get_dataloader(config, dataset, mode='train', multi_process=False):
    """Get dataloader with config, dataset, mode as input, allows multiGPU settings.

        Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: see config.py for details
        dataset: paddle.io.dataset object
        mode: train/val
        multi_process: if True, use DistributedBatchSampler to support multi-processing
    Returns:
        dataloader: paddle.io.DataLoader object.
    """

    if mode == 'train':
        batch_size = config.DATA.BATCH_SIZE
    else:
        batch_size = config.DATA.BATCH_SIZE_EVAL

    if multi_process is True:
        sampler = DistributedBatchSampler(dataset,
                                          batch_size=batch_size,
                                          shuffle=(mode == 'train'))
        dataloader = DataLoader(dataset,
                                batch_sampler=sampler,
                                num_workers=config.DATA.NUM_WORKERS)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=(mode == 'train'))
    return dataloader
