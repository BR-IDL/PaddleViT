import os
import glob
from src.transforms import Compose
from paddle.io import Dataset
from PIL import Image
import numpy as np
import random


class Trans10kV2(Dataset):
    """Trans10kV2 
    
    It contains the first extensive transparent object segmentation dataset,
    which contains 11 fine-grained transparent object categories

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Trans10kV2 dataset directory.
        mode (str, optional): Which part of dataset to use. Default: 'train'.
        num_classes (int): the number of classes
    """

    def __init__(self, config, transforms, dataset_root, mode='train', num_classes=12, ignore_index=255):
        super(Trans10kV2, self).__init__()
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.file_list = list()
        self.crop_size = config.DATA.CROP_SIZE
        self.resize = config.DATA.AUG.RESIZE
        self.mirror = config.DATA.AUG.MIRROR
        self.ignore_index = ignore_index
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        if mode == 'val':
            mode = 'validation'
            self.transforms = Compose(transforms)
        img_dir = os.path.join(self.dataset_root, mode, 'images')
        label_dir = os.path.join(self.dataset_root, mode, 'masks_12')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError("The dataset is not Found or the folder structure" 
                             "is nonconfoumance.")

        label_files = sorted(glob.glob(os.path.join(label_dir, '*_mask.png')))
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]
        print("mode: {}, file_nums: {}".format(mode, len(self.file_list)))

    def _sync_transform(self, img, label, resize=False, mirror=False):
        if resize:
            img = img.resize(self.crop_size, Image.BILINEAR)
            label = label.resize(self.crop_size, Image.NEAREST)
        if mirror and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        img, label = np.array(img), np.array(label).astype('int64')
        return img, label
    
    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        img = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert("P")
        if self.mode == 'test':
            img, _ = self.transforms(img=image_path)
            img = img[np.newaxis, ...]
            return img, image_path
        elif self.mode == 'val':
            img, _ = self.transforms(img=image_path)
            label = np.asarray(Image.open(label_path).convert('P'))
            label = label[np.newaxis, :, :]
            return img, label
        else:
            img, label = self._sync_transform(img, label, resize=self.resize, mirror=self.mirror)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.file_list)
