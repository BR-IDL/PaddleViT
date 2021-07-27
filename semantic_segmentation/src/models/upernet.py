import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import copy
import numpy as np
import math
import sys
from src.models.backbones import SwinTransformer
from src.models.decoders import UperHead, FCNHead
from src.utils.utils import load_pretrained_model


class UperNet(nn.Layer):
    """ UperNet

    Reference:
        Tete Xiao, et al. *"Unified Perceptual Parsing for Scene Understanding"*
    """
    def __init__(self, config):
        super(UperNet, self).__init__()
        if config.MODEL.ENCODER.TYPE == "SwinTransformer":
            self.encoder = SwinTransformer(config)
        self.num_layers = len(config.MODEL.TRANS.STAGE_DEPTHS)
        self.AuxiHead = config.MODEL.AUX.AUXIHEAD
        self.decoder_type = config.MODEL.DECODER_TYPE
        self.backbone_out_indices = config.MODEL.ENCODER.OUT_INDICES

        assert self.decoder_type == "UperHead", "only support UperHead decoder"

        self.num_features = [int(config.MODEL.TRANS.EMBED_DIM * 2 ** i) for i in range(self.num_layers)]

        self.layer_norms = nn.LayerList()
        for idx in self.backbone_out_indices:
           self.layer_norms.append(nn.LayerNorm(self.num_features[idx]))

        self.decoder = UperHead(
            pool_scales = config.MODEL.UPERHEAD.POOL_SCALES, 
            in_channels = config.MODEL.UPERHEAD.IN_CHANNELS,
            channels = config.MODEL.UPERHEAD.CHANNELS,
            align_corners = config.MODEL.UPERHEAD.ALIGN_CORNERS,
            num_classes = config.DATA.NUM_CLASSES)
        self.AuxiHead = config.MODEL.AUX.AUXIHEAD
        if self.AuxiHead==True:
            self.aux_decoder = FCNHead(
                in_channels = config.MODEL.AUXFCN.IN_CHANNELS, 
                num_classes = config.DATA.NUM_CLASSES, 
                up_ratio = config.MODEL.AUXFCN.UP_RATIO) 
        #self.init__decoder_lr_coef(config)
    
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
            print("self.aux_decoders.sublayers(): ", sublayers)
            for sublayer_list in sublayers:
                for sublayer in sublayer_list:
                    if isinstance(sublayer, nn.Conv2D):
                        #print("sublayer: ", sublayer)
                        sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                        if sublayer.bias is not None:
                            sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF


    def to_2D(self, x):
        n, hw, c = x.shape                                                                                                                                    
        h = w = int(math.sqrt(hw))
        x = x.transpose([0, 2, 1]).reshape([n, c, h, w])
        return x

    def forward(self, imgs):
        # imgs.shapes: (B,3,H,W)
        feats = self.encoder(imgs)
        for idx in self.backbone_out_indices:
            feat = self.layer_norms[idx](feats[idx])
            feats[idx] = self.to_2D(feat)
        p2, p3, p4, p5 = feats
        preds = [self.decoder([p2, p3, p4, p5]), ]
        preds.append(self.aux_decoder(p4))
        return preds

