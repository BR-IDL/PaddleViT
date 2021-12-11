#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""                                                                                                                                                                                                                 
This module implements SETR
Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers
<https://arxiv.org/pdf/2012.15840.pdf>
"""

import paddle
import paddle.nn as nn
from src.models.backbones import ViT_MLA, VisualTransformer
from src.models.decoders import VIT_MLAHead, VIT_MLA_AUXIHead, VisionTransformerUpHead
from src.utils import load_pretrained_model


class SETR(nn.Layer):
    """ SETR

    SEgmentation TRansformer (SETR) has three diffrent decoder designs to 
    perform pixl-level segmentation. The variants of SETR includes SETR_MLA, 
    SETR_PUP, and SETR_Naive.

    Attributes:
        encoder: A backbone network for extract features from image.
        auxi_head: A boolena indicating if we employ the auxilary segmentation head.
        decoder_type: Type of decoder.
        decoder: A decoder module for semantic segmentation.
    """
    def __init__(self, config):
        super(SETR, self).__init__()
        if config.MODEL.ENCODER.TYPE == "ViT_MLA":
            self.encoder = ViT_MLA(config)
        elif config.MODEL.ENCODER.TYPE == "ViT":
            self.encoder = VisualTransformer(config)
        self.auxi_head = config.MODEL.AUX.AUXIHEAD
        self.decoder_type = config.MODEL.DECODER_TYPE

        if self.decoder_type == "VIT_MLAHead":
            self.decoder = VIT_MLAHead(
                config.MODEL.MLA.MLA_CHANNELS,
                config.MODEL.MLA.MLAHEAD_CHANNELS, 
                config.DATA.NUM_CLASSES,
                config.MODEL.MLA.MLAHEAD_ALIGN_CORNERS) 
            self.auxi_head = config.MODEL.AUX.AUXIHEAD
            if self.auxi_head == True:
                self.aux_decoder2 = VIT_MLA_AUXIHead(
                    config.MODEL.MLA.MLA_CHANNELS, 
                    config.DATA.NUM_CLASSES, 
                    config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
                self.aux_decoder3 = VIT_MLA_AUXIHead(
                    config.MODEL.MLA.MLA_CHANNELS, 
                    config.DATA.NUM_CLASSES, 
                    config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
                self.aux_decoder4 = VIT_MLA_AUXIHead(
                    config.MODEL.MLA.MLA_CHANNELS, 
                    config.DATA.NUM_CLASSES, 
                    config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 
                self.aux_decoder5 = VIT_MLA_AUXIHead(
                    config.MODEL.MLA.MLA_CHANNELS, 
                    config.DATA.NUM_CLASSES, 
                    config.MODEL.AUX.AUXHEAD_ALIGN_CORNERS) 

        elif (self.decoder_type == "PUP_VisionTransformerUpHead" or 
              self.decoder_type == "Naive_VisionTransformerUpHead"):
            self.decoder = VisionTransformerUpHead(
                config.MODEL.PUP.INPUT_CHANNEL, 
                config.MODEL.PUP.NUM_CONV, 
                config.MODEL.PUP.NUM_UPSAMPLE_LAYER, 
                config.MODEL.PUP.CONV3x3_CONV1x1, 
                config.MODEL.PUP.ALIGN_CORNERS, 
                config.DATA.NUM_CLASSES)
            if self.auxi_head == True:
                self.aux_decoder2 = VisionTransformerUpHead(
                    config.MODEL.AUXPUP.INPUT_CHANNEL, 
                    config.MODEL.AUXPUP.NUM_CONV, 
                    config.MODEL.AUXPUP.NUM_UPSAMPLE_LAYER, 
                    config.MODEL.AUXPUP.CONV3x3_CONV1x1, 
                    config.MODEL.AUXPUP.ALIGN_CORNERS, 
                    config.DATA.NUM_CLASSES)
                self.aux_decoder3 = VisionTransformerUpHead(
                    config.MODEL.AUXPUP.INPUT_CHANNEL, 
                    config.MODEL.AUXPUP.NUM_CONV,  
                    config.MODEL.AUXPUP.NUM_UPSAMPLE_LAYER, 
                    config.MODEL.AUXPUP.CONV3x3_CONV1x1, 
                    config.MODEL.AUXPUP.ALIGN_CORNERS, 
                    config.DATA.NUM_CLASSES)
                self.aux_decoder4 = VisionTransformerUpHead(
                    config.MODEL.AUXPUP.INPUT_CHANNEL, 
                    config.MODEL.AUXPUP.NUM_CONV,
                    config.MODEL.AUXPUP.NUM_UPSAMPLE_LAYER, 
                    config.MODEL.AUXPUP.CONV3x3_CONV1x1, 
                    config.MODEL.AUXPUP.ALIGN_CORNERS, 
                    config.DATA.NUM_CLASSES)
                if self.decoder_type == "PUP_VisionTransformerUpHead":
                    self.aux_decoder5 = VisionTransformerUpHead(
                        config.MODEL.AUXPUP.INPUT_CHANNEL, 
                        config.MODEL.AUXPUP.NUM_CONV,
                        config.MODEL.AUXPUP.NUM_UPSAMPLE_LAYER, 
                        config.MODEL.AUXPUP.CONV3x3_CONV1x1, 
                        config.MODEL.AUXPUP.ALIGN_CORNERS, 
                        config.DATA.NUM_CLASSES)
        self.init__decoder_lr_coef(config)
    
    def init__decoder_lr_coef(self, config):
        #print("self.decoder.sublayers(): ", self.decoder.sublayers())
        for sublayer in self.decoder.sublayers():
            #print("F sublayer: ", sublayer)
            if isinstance(sublayer, nn.Conv2D):
                #print("sublayer: ", sublayer)
                sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                if sublayer.bias is not None:
                    sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
            if (isinstance(sublayer, nn.SyncBatchNorm) or 
               isinstance(sublayer, nn.BatchNorm2D) or 
                isinstance(sublayer,nn.LayerNorm)):
                #print("SyncBN, BatchNorm2D, or LayerNorm")
                #print("sublayer: ", sublayer)
                sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
        if self.auxi_head == True:
            sublayers = []  # list of list
            sublayers.append(self.aux_decoder2.sublayers())
            sublayers.append(self.aux_decoder3.sublayers())
            sublayers.append(self.aux_decoder4.sublayers())
            if self.decoder_type == "PUP_VisionTransformerUpHead":
                 sublayers.append(self.aux_decoder5.sublayers())
            #print("self.aux_decoders.sublayers(): ", sublayers)
            for sublayer_list in sublayers:
                for sublayer in sublayer_list:
                    if isinstance(sublayer, nn.Conv2D):
                        #print("sublayer: ", sublayer)
                        sublayer.weight.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF
                        if sublayer.bias is not None:
                            sublayer.bias.optimize_attr['learning_rate'] = config.TRAIN.DECODER_LR_COEF


    def forward(self, imgs):
        # imgs.shapes: (B,3,H,W)
        p2, p3, p4, p5 = self.encoder(imgs)
        preds = []
        if self.decoder_type == "VIT_MLAHead":
            pred = self.decoder(p2, p3, p4, p5)
        elif (self.decoder_type == "PUP_VisionTransformerUpHead" or 
              self.decoder_type == "Naive_VisionTransformerUpHead"):
            pred = self.decoder(p5)
        preds.append(pred)
        if self.auxi_head == True:
            preds.append(self.aux_decoder2(p2))
            preds.append(self.aux_decoder3(p3))
            preds.append(self.aux_decoder4(p4))
            if self.decoder_type == "PUP_VisionTransformerUpHead":
                preds.append(self.aux_decoder5(p5))
        return preds

