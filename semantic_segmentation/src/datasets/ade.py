import os
import numpy as np
from PIL import Image
from src.datasets import Dataset
from src.utils.download import download_file_and_uncompress
from src.utils import seg_env
from src.transforms import Compose
import src.transforms.functional as F

URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


class ADE20K(Dataset):
    """
    ADE20K dataset `http://sceneparsing.csail.mit.edu/`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
    """

    def __init__(self, transforms, dataset_root=None, mode='train', num_classes=150):
        super(ADE20K, self).__init__(transforms=transforms, num_classes=num_classes,
            dataset_root=dataset_root, mode=mode)
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = num_classes
        self.ignore_index = 255

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in ADE20K dataset, but got {}.".format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME,
                extraname='ADEChallengeData2016')
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        if mode == 'train':
            img_dir = os.path.join(self.dataset_root, 'images/training')
            label_dir = os.path.join(self.dataset_root, 'annotations/training')
        elif mode == 'val':
            img_dir = os.path.join(self.dataset_root, 'images/validation')
            label_dir = os.path.join(self.dataset_root,
                                     'annotations/validation')
        img_files = os.listdir(img_dir)
        label_files = [i.replace('.jpg', '.png') for i in img_files]
        for i in range(len(img_files)):
            img_path = os.path.join(img_dir, img_files[i])
            label_path = os.path.join(label_dir, label_files[i])
            self.file_list.append([img_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            label = label - 1
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            label = label - 1
            # Recover the ignore pixels adding by transform
            label[label == 254] = 255
            return im, label
