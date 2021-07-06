
import os

from PIL import Image
from src.datasets import Dataset
from src.transforms import Compose


class PascalContext(Dataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories.  The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str): Which part of dataset to use. it is one of ('train', 'trainval', 'context', 'val').
            If you want to set mode to 'context', please make sure the dataset have been augmented. Default: 'train'.
    """

    def __init__(self, transforms=None, dataset_root=None, mode='train', num_classes=60):
        super(PascalContext, self).__init__(transforms=transforms, num_classes=num_classes, 
            dataset_root=dataset_root, mode=mode)

        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = num_classes
        self.ignore_index = 255


        if mode not in ['train', 'trainval', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'val') in PascalContext dataset, but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if self.dataset_root is None:
            raise ValueError("The dataset is not Found or the folder structure is nonconfoumance.")
        
        image_set_dir = os.path.join(self.dataset_root, 'ImageSets', 'SegmentationContext')

        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val.txt')
            #file_path = os.path.join(image_set_dir, 'val_mini.txt')
        elif mode == 'trainval':
            file_path = os.path.join(image_set_dir, 'trainval.txt')
        print("file_path: ", file_path)
        if not os.path.exists(file_path):
            raise RuntimeError("PASCAL-Context annotations are not ready, ")

        img_dir = os.path.join(self.dataset_root, 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'SegmentationClassContext')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])
        print("mode: {}, file_nums: {}".format(mode, len(self.file_list)))
