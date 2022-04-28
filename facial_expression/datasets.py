import os
import math
import numpy as np
import random
import glob
import PIL
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.vision import transforms
from paddle.vision import image_load
from random_erasing import RandomErasing
from config import get_config


class ABAWDataset(Dataset):
    def __init__(self, file_folder, anno_folder, data_list=None, class_type='all', is_train=True, transform_ops=None):
        super().__init__()
        assert class_type in ['all', 'coarse', 'negative']
        anno_folder = os.path.join(anno_folder, 'Train_Set' if is_train else "Validation_Set")
        class_names_original = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        class_names_coarse = ['Neutral', 'Happiness', 'Surprise', 'Other', 'Negative']
        class_names_negative = ['Anger', 'Disgust', 'Fear', 'Sadness']
        class_mapping = {
            'all': None,
            'coarse': [0, 4, 4, 4, 1, 4, 2, 3],
            'negative': [-1, 0, 1, 2, -1, 3, -1, -1]
        }
        self.transforms = transform_ops
        self.file_folder = file_folder

        if data_list is not None and os.path.isfile(data_list):
            print(f'----- Loading data list form: {data_list}')
            self.data_list = []
            with open(data_list, 'r') as infile:
                for line in infile:
                    self.data_list.append(
                        (line.split(' ')[0], int(line.split(' ')[1]))
                    )
        else:
            print(f'----- Generating data list form: {anno_folder}')
            save_path =  f'./train_list_{class_type}.txt' if is_train else f'./val_list_{class_type}.txt' 
            self.data_list = self.gen_list(file_folder,
                                           anno_folder,
                                           class_mapping=class_mapping[class_type],
                                           save_path=save_path)
        print(f'----- Total images: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = image_load(os.path.join(self.file_folder, self.data_list[index][0])).convert('RGB')
        data = self.transforms(data)
        label = self.data_list[index][1]
        image_path = self.data_list[index][0]

        return data, label, image_path

    def gen_list(self, file_folder, anno_folder, class_mapping=None, save_path=None):
        """Generate list of data samples where each line contains image path and its label
            Input:
                file_folder: folder path of images (aligned)
                anno_folder: folder path of annotations, e.g., ./EXPR_Classification_Challenge/
                class_mapping: list, class mapping for negative and coarse
                save_path: path of a txt file for saving list, default None
            Output:
                out_list: list of tuple contains relative file path and its label 
        """
        out_list = []
        for label_file in glob.glob(os.path.join(anno_folder, '*.txt')):
            with open(label_file, 'r') as infile:
                print(f'----- Reading labels from: {os.path.basename(label_file)}')
                vid_name = os.path.basename(label_file)[0:-4]
                for idx, line in enumerate(infile):
                    if idx == 0:
                        classnames = line.split(',')
                    else:
                        label = int(line)
                        if label == -1: # eliminate data with -1 label
                            continue
                        if class_mapping is not None:
                            label = class_mapping[label]
                            if label == -1: # eliminate data with -1 label (negative)
                                continue

                        image_name = f'{str(idx).zfill(5)}.jpg'
                        if os.path.isfile(os.path.join(file_folder, vid_name, image_name)):
                            out_list.append((os.path.join(vid_name, image_name), label)) # tuple
        if save_path is not None:
            with open(save_path, 'w') as ofile:
                for path, label in out_list:
                    ofile.write(f'{path} {label}\n')
            print(f'List saved to: {save_path}')

        return out_list
            

class RandomApply():
    def __init__(self, transforms, prob=0.5):
        self.prob = prob
        self.transforms = transforms
    def __call__(self, x):
        if random.random() > self.prob:
            for t in self.transforms:
                x = t(x)
        return x

class GaussianBlur():
    def __init__(self, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_train_transforms(config):
    aug_op_list = []
    aug_op_list.append(RandomApply([transforms.RandomRotation(degrees=6)], prob=0.5))
    aug_op_list.append(
        transforms.RandomResizedCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE),
        scale=(0.08, 1.0), ratio=(1., 1.), interpolation='bicubic'))
    aug_op_list.append(transforms.RandomHorizontalFlip())
    aug_op_list.append(RandomApply([transforms.Grayscale()], prob=0.2))
    aug_op_list.append(RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4)], prob=0.8))
    aug_op_list.append(RandomApply([GaussianBlur(0.1, 2.0)], prob=0.5))
    aug_op_list.append(transforms.ToTensor())
    aug_op_list.append(transforms.Normalize(mean=config.DATA.IMAGENET_MEAN,
                                            std=config.DATA.IMAGENET_STD))
    if config.TRAIN.RANDOM_ERASE_PROB > 0.:
        random_erasing = RandomErasing(prob=config.TRAIN.RANDOM_ERASE_PROB,
                                       mode=config.TRAIN.RANDOM_ERASE_MODE,
                                       max_count=config.TRAIN.RANDOM_ERASE_COUNT,
                                       num_splits=config.TRAIN.RANDOM_ERASE_SPLIT)
        aug_op_list.append(random_erasing)
    transforms_train = transforms.Compose(aug_op_list)
    return transforms_train


def get_val_transforms(config):
    scale_size = int(math.floor(config.DATA.IMAGE_SIZE / config.DATA.CROP_PCT))
    transforms_val = transforms.Compose([
        transforms.Resize(scale_size, 'bicubic'), # single int for resize shorter side of image
        transforms.CenterCrop((config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.DATA.IMAGENET_MEAN, std=config.DATA.IMAGENET_STD)])
    return transforms_val


def get_dataset(config, is_train=True):
    if config.DATA.DATASET == "ABAW":
        if is_train:
            transform_ops = get_train_transforms(config)
        else:
            transform_ops = get_val_transforms(config)
        dataset = ABAWDataset(file_folder=config.DATA.DATA_FOLDER,
                              anno_folder=config.DATA.ANNO_FOLDER,
                              data_list=config.DATA.DATA_LIST_TRAIN if is_train else config.DATA.DATA_LIST_VAL,
                              class_type=config.DATA.CLASS_TYPE,
                              is_train=is_train,
                              transform_ops=transform_ops)
    else:
        raise NotImplementedError(
            "Wrong dataset name: [{config.DATA.DATASET}]. Only 'imagenet2012' is supported now")
    return dataset


def get_dataloader(config, dataset, is_train=True, use_dist_sampler=False):
    """Get dataloader from dataset, allows multiGPU settings.
    Multi-GPU loader is implements as distributedBatchSampler.

    Args:
        config: see config.py for details
        dataset: paddle.io.dataset object
        is_train: bool, when False, shuffle is off and BATCH_SIZE_EVAL is used, default: True
        use_dist_sampler: if True, DistributedBatchSampler is used, default: False
    Returns:
        dataloader: paddle.io.DataLoader object.
    """
    batch_size = config.DATA.BATCH_SIZE if is_train else config.DATA.BATCH_SIZE_EVAL

    if use_dist_sampler is True:
        sampler = DistributedBatchSampler(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=is_train,
                                          drop_last=is_train)
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                num_workers=config.DATA.NUM_WORKERS)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=is_train,
                                drop_last=is_train)
    return dataloader



def main():
    config = get_config()
    # train dataset
    #transform_ops = get_train_transforms(config)
    #dataset = ABAWDataset(file_folder='./abaw_dataset/aligned_MTCNNXue/',
    #                      anno_folder='./abaw_dataset/Third_ABAW_Annotations/EXPR_Classification_Challenge/Train_Set',
    #                      #data_list='./train_list_all.txt',
    #                      data_list=None,
    #                      class_type='negative',
    #                      is_train=True,
    #                      transform_ops=transform_ops)
    # val dataset
    transform_ops = get_val_transforms(config)
    dataset = ABAWDataset(file_folder='./abaw_dataset/aligned_MTCNNXue/',
                          anno_folder='./abaw_dataset/Third_ABAW_Annotations/EXPR_Classification_Challenge',
                          data_list=None,
                          class_type='coarse',
                          is_train=False,
                          transform_ops=transform_ops)

    for idx, sample in enumerate(dataset):
        if idx == 10:
            break
        print(sample[0], sample[1])

if __name__ == "__main__":
    main()
