import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import copy
import numpy as np
import math
import sys
from src.models.backbones import ViT_MLA
from src.models.decoders import VIT_MLAHead, VIT_MLA_AUXIHead
from src.utils.utils import load_pretrained_model


class SETR_MLA(nn.Layer):
    """ SETR_MLA
    
    Reference:
        Sixiao Zheng, et al. *"Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"*
    """
    def __init__(self, config):
        super(SETR_MLA, self).__init__()
        self.encoder = ViT_MLA(config)
        self.decoder = VIT_MLAHead(config.MODEL.MLA.MLA_CHANNELS,
                                   config.MODEL.MLA.MLAHEAD_CHANNELS, 
                                   config.DATA.NUM_CLASSES,
                                   config.MODEL.MLA.MLAHEAD_ALIGN_CORNERS) 
        self.AuxiHead = config.MODEL.AUX.AUXIHEAD
        if self.AuxiHead==True:
            self.aux_decoder2 = VIT_MLA_AUXIHead(config.MODEL.MLA.MLA_CHANNELS, config.DATA.NUM_CLASSES, config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
            self.aux_decoder3 = VIT_MLA_AUXIHead(config.MODEL.MLA.MLA_CHANNELS, config.DATA.NUM_CLASSES, config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
            self.aux_decoder4 = VIT_MLA_AUXIHead(config.MODEL.MLA.MLA_CHANNELS, config.DATA.NUM_CLASSES, config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
            self.aux_decoder5 = VIT_MLA_AUXIHead(config.MODEL.MLA.MLA_CHANNELS, config.DATA.NUM_CLASSES, config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 

        self.init__decoder_lr_coef(config)
    
    def init__decoder_lr_coef(self,config):
        #print("self.decoder.sublayers(): ", self.decoder.sublayers())
        for sublayer in self.decoder.sublayers():
            #print("F sublayer: ", sublayer)
            if isinstance(sublayer, nn.Conv2D):
                print("sublayer: ", sublayer)
                sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                if sublayer.bias is not None:
                    sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
            if isinstance(sublayer, nn.SyncBatchNorm) or isinstance(sublayer, nn.BatchNorm2D) or isinstance(sublayer,nn.LayerNorm):
                #print("SyncBN, BatchNorm2D, or LayerNorm")
                print("sublayer: ", sublayer)
                sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
        if self.AuxiHead==True:
            sublayers = []  # list of list
            sublayers.append(self.aux_decoder2.sublayers())
            sublayers.append(self.aux_decoder3.sublayers())
            sublayers.append(self.aux_decoder4.sublayers())
            sublayers.append(self.aux_decoder5.sublayers())
            print("self.aux_decoders.sublayers(): ", sublayers)
            for sublayer_list in sublayers:
                for sublayer in sublayer_list:
                    if isinstance(sublayer, nn.Conv2D):
                        #print("sublayer: ", sublayer)
                        sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                        if sublayer.bias is not None:
                            sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF


    def forward(self, imgs):
        # imgs.shapes: (B,3,H,W)
        mla_p2, mla_p3, mla_p4, mla_p5 = self.encoder(imgs)
        pred = self.decoder(mla_p2, mla_p3, mla_p4, mla_p5)
        if self.AuxiHead==True:
            aux_pred2 = self.aux_decoder2(mla_p2)
            aux_pred3 = self.aux_decoder3(mla_p3)
            aux_pred4 = self.aux_decoder4(mla_p4)
            aux_pred5 = self.aux_decoder5(mla_p5)
        if self.AuxiHead==True:
            return [pred, aux_pred2, aux_pred3, aux_pred4, aux_pred5]
        return [pred]

