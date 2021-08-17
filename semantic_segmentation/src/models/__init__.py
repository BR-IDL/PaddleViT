from .setr import SETR
from .upernet import UperNet
from .dpt import DPTSeg
from .segmentor import Segmentor


def get_model(config):
    if "SETR" in config.MODEL.NAME:
       model = SETR(config)
    elif "UperNet" in config.MODEL.NAME:
       model = UperNet(config)
    elif "DPT" in config.MODEL.NAME:
       model = DPTSeg(config)
    elif "Segmenter" in config.MODEL.NAME:
       model = Segmentor(config)
    return model
