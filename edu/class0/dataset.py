from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision import transforms


def get_transforms(mode='train'):
    if mode == 'train':
        data_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    return data_transforms


def get_dataset(name='cifar10', mode='train'):
    if name == 'cifar10':
        dataset = datasets.Cifar10(mode=mode, transform=get_transforms(mode))

    return dataset

def get_dataloader(dataset, batch_size=128, mode='train'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=(mode == 'train'))
    return dataloader


