import os
import paddle
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms, datasets, image_load, set_image_backend
import numpy as np
import argparse
from PIL import Image
import cv2
from config import *

class ImageNet1MDataset(Dataset):
    def __init__(self, file_folder, mode="train", transform=None):
        super(ImageNet1MDataset, self).__init__()
        assert mode in ["train", "val"]
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = []
        self.label_list = []

        if mode=="train":
            self.list_file = os.path.join(self.file_folder, "train_list.txt")
        else:
            self.list_file = os.path.join(self.file_folder, "val_list.txt")

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder,img_path))
                self.label_list.append(img_label)
        print(len(self.label_list))

    def __len__(self):
        return len(self.label_list)
        
    def __getitem__(self, index):
        #print(self.img_path_list[index])
        #if os.path.isfile(self.img_path_list[index]):
        #    print('exist')
        #else:
        #    print('not exist')
        #data = Image.open(self.img_path_list[index]).convert('L')
        #data = cv2.imread(self.img_path_list[index])
        set_image_backend('cv2')
        data = image_load(self.img_path_list[index])
        data = self.transform(data)
        label = self.label_list[index] 

        return data, label
        

def get_dataset(config):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) 

    transform_test = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) 

    if config.DATASET == "cifar10":
        dataset_train = datasets.Cifar10(mode="train", transform=transform_train)
        dataset_test = datasets.Cifar10(mode="test", transform=transform_test)
    elif config.DATASET == "cifar100":
        dataset_train = datasets.Cifar100(mode="train", transform=transform_train)
        dataset_test = datasets.Cifar100(mode="test", transform=transform_test)
    elif config.DATASET == "imagenet1m":
        dataset_train = ImageNet1MDataset(config.DATA_PATH, mode="train", transform=transform_train)
        dataset_test = ImageNet1MDataset(config.DATA_PATH, mode="val", transform=transform_test)
    else:
        raise NotImplementedError("Only cifar10, cifar100, imagenet1m are supported now")

    return dataset_train, dataset_test


def get_loader(config, dataset_train, dataset_test=None, multi=False):
    # multigpu 
    if multi:
        sampler_train = paddle.io.DistributedBatchSampler(dataset_train,
                                                          batch_size=config.BATCH_SIZE,
                                                          shuffle=True,
                                                          )
        dataloader_train = DataLoader(dataset_train, batch_sampler=sampler_train)
        if dataset_test is not None:
            sampler_test = paddle.io.DistributedBatchSampler(dataset_test,
                                                             batch_size=config.BATCH_SIZE,
                                                             shuffle=False,
                                                            )
            dataloader_test = DataLoader(dataset_test, batch_sampler=sampler_test)
        else:
            dataloader_test = None
    else:
    # single gpu
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=config.BATCH_SIZE,
                                      num_workers=config.NUM_WORKERS,
                                      shuffle=True,
                                      #places=paddle.CUDAPlace(0),
                                      )

        if dataset_test is not None:
            dataloader_test = DataLoader(dataset_test,
                                         batch_size=config.BATCH_SIZE,
                                         num_workers=config.NUM_WORKERS,
                                         shuffle=False,
                                         #places=paddle.CUDAPlace(0),
                                         )
        else:
            dataloader_test = None


    return dataloader_train, dataloader_test




def main():
    print('dataset and dataloader')
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default="imagenet1m")
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-image_size', type=int, default=224)
    parser.add_argument('-data_path', type=str, default='/dataset/imagenet/')
    args = parser.parse_args()
    print(args)

    config = get_config()
    config = update_config(config, args)
    print(config)

    dt_trn, dt_tst = get_dataset(config.DATA)
    dl_trn, dl_tst = get_loader(config.DATA, dt_trn, dt_tst)

    for idx, (batch_data, batch_label) in enumerate(dl_tst):
        print(batch_data.shape)
        print(batch_label)
        if idx == 10:
            break



if __name__ == "__main__":
    main()
