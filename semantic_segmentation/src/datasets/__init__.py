from .dataset import Dataset
from .cityscapes import Cityscapes
from .ade import ADE20K
from .pascal_context import PascalContext


def get_dataset(config, data_transform, mode='train'):

    if config.DATA.DATASET == "PascalContext":
        if mode == 'train':
            dataset = PascalContext(transforms=data_transform,
                dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES, mode='train')
        elif mode == 'val':
            dataset = PascalContext(transforms=data_transform, 
                dataset_root=config.DATA.DATA_PATH,num_classes=config.DATA.NUM_CLASSES, mode='val')
    elif config.DATA.DATASET == "Cityscapes":
        if mode == 'train':
            dataset = Cityscapes(transforms=data_transform,
                dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES, mode='train')
        elif mode == 'val':
            dataset =  Cityscapes(transforms=data_transform,
                dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES, mode='val')
    elif config.DATA.DATASET == "ADE20K":
        if mode == 'train':
            dataset = ADE20K(transforms=data_transform,                                                                                                                                                                                   
                dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES, mode='train')
        elif mode == 'val':
            dataset =  ADE20K(transforms=data_transform,
                dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES, mode='val')
    else:
        raise NotImplementedError("{} dataset is not supported".format(config.DATA.DATASET))

    return dataset
