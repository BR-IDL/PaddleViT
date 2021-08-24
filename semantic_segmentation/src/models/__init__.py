from .setr import SETR
from .upernet import UperNet
from .dpt import DPTSeg
from .segmentor import Segmentor
from .trans2seg import Trans2Seg
from .segformer import Segformer


def get_model(config):
    if "SETR" in config.MODEL.NAME:
       model = SETR(config)
    elif "UperNet" in config.MODEL.NAME:
       model = UperNet(config)
    elif "DPT" in config.MODEL.NAME:
       model = DPTSeg(config)
    elif "Segmenter" in config.MODEL.NAME:
       model = Segmentor(config)
    elif 'Trans2Seg' in config.MODEL.NAME:
       model = Trans2Seg(config)
    elif "Segformer" in config.MODEL.NAME:
       model = Segformer(config)
    return model
