from .transforms import *
from . import functional

def get_transforms(config):
    if config.DATA.DATASET == "Trans10kV2":
        transforms_train = [Resize(target_size=config.DATA.CROP_SIZE),
                            RandomHorizontalFlip(prob=0.5),
                            Normalize(mean=[123.675, 116.28, 103.53],
                                      std=[58.395, 57.12, 57.375])]
    elif config.DATA.DATASET == "ADE20K":
        transforms_train = [ResizeStepScaling(min_scale_factor=0.5, 
                                              max_scale_factor=2.0, 
                                              scale_step_size=0.25),
                            RandomPaddingCrop(crop_size=config.DATA.CROP_SIZE, 
                                              img_padding_value=(123.675, 116.28, 103.53), 
                                              label_padding_value=255),
                            RandomHorizontalFlip(prob=0.5),
                            RandomDistort(brightness_range=0.4, 
                                          contrast_range=0.4, 
                                          saturation_range=0.4),
                            Normalize(mean=[123.675, 116.28, 103.53],
                                      std=[58.395, 57.12, 57.375])]
    else:
        raise NotImplementedError("{} dataset is not supported".format(config.DATA.DATASET))
    return transforms_train
