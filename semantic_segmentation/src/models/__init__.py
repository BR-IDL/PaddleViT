from .setr import SETR
from .upernet import UperNet
from .dpt import DPTSeg


def get_model(config):
    if "SETR" in config.MODEL.NAME:
       model = SETR(config)
    elif "UperNet" in config.MODEL.NAME:
       model = UperNet(config)
    return model
