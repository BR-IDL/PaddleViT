from .dataset import Dataset
from .cityscapes import Cityscapes
from .ade import ADE20K
from .pascal_context import PascalContext


def get_dataset(config):

    if config.DATA.DATASET == "PascalContext":
        dataset_train = PascalContext(transforms=config.DATASET.Train_Pipeline,
                      dataset_root=config.DATA.DATA_PATH, num_classes=config.DATA.NUM_CLASSES,mode='train')
        dataset_test = PascalContext(transforms=config.DATASET.Test_Pipeline, 
                      dataset_root=config.DATA.DATA_PATH,num_classes=config.DATA.NUM_CLASSES, mode='val')
    else:
        raise NotImplementedError("Only PascalContext are supported now")

    return dataset_train, dataset_test
