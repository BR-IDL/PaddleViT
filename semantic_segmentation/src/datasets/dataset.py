import os
import paddle
import numpy as np
from PIL import Image
from src.transforms import Compose
import src.transforms.functional as F


class Dataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            image.jpg label.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        ignore_index (int): ignore label, default=255

    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes,
                 mode='train',
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(self.dataset_root))

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            '''
            normalize = paddle.vision.transforms.Normalize(mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        data_format='HWC')
            im = np.asarray(normalize(Image.open(image_path).resize((512, 512), Image.BILINEAR))).transpose(2, 0, 1)
            label = np.asarray(Image.open(label_path).resize((512, 512), Image.BILINEAR))
            return im, label
            '''
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path).convert('P'))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            
            return im, label

    def __len__(self):
        return len(self.file_list)
