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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from src.models.decoders.fcn_head import FCNHead
from src.models.decoders import ConvBNReLU, SeparableConv2d, CNNHEAD, HybridEmbed
from .backbones import get_segmentation_backbone, TransformerEncoder, TransformerDecoder, expand


class Trans2Seg(nn.Layer):
    """Trans2Seg Implement
    
    It contains cnn-encoder, transformer-encoder and transformer-decoder, and a small-cnn-head
    Ref, https://arxiv.org/pdf/2101.08461.pdf

    """
    def __init__(self, config):
        super(Trans2Seg, self).__init__()
        c1_channels = 256
        c4_channels = 2048
        self.nclass = config.DATA.NUM_CLASSES
        self.aux = config.MODEL.AUX.AUXIHEAD
        self.backbone = config.MODEL.ENCODER.TYPE.lower()
        
        # Create cnn encoder, the input image is fed to CNN to extract features
        self.cnn_encoder = get_segmentation_backbone(self.backbone, config, nn.BatchNorm2D)
        
        # Get vit hyper params
        vit_params = config.MODEL.TRANS2SEG
        hid_dim = config.MODEL.TRANS2SEG.HID_DIM

        c4_HxW = (config.DATA.CROP_SIZE[0] // 16) ** 2
        vit_params['decoder_feat_HxW'] = c4_HxW

        last_channels = vit_params['EMBED_DIM']

        # create transformer encoder, for transformer encoder,
        # the features and position embedding are flatten and fed to transformer for self-attention,
        # and output feature(Fe) from transformer encoder.
        self.transformer_encoder = TransformerEncoder(
                                     embed_dim=last_channels,
                                     depth=vit_params['DEPTH'],
                                     num_heads=vit_params['NUM_HEADS'],
                                     mlp_ratio=vit_params['MLP_RATIO'])
        # create transformer decoder, for transformer decoder,
        # for transformer decoder, we specifically define a set of learnable class prototype embeddings as query,
        # the features from transformer encoder as key
        self.transformer_decoder = TransformerDecoder(
                                     nclass=config.DATA.NUM_CLASSES,
                                     embed_dim=last_channels,
                                     depth=vit_params['DEPTH'],
                                     num_heads=vit_params['NUM_HEADS'],
                                     mlp_ratio=vit_params['MLP_RATIO'],
                                     decoder_feat_HxW=vit_params['decoder_feat_HxW'])
        # Create Hybrid Embedding
        self.hybrid_embed = HybridEmbed(c4_channels, last_channels)
        # Create small Conv Head, a small conv head to fuse attention map and Res2 feature from CNN backbone
        self.cnn_head = CNNHEAD(vit_params, c1_channels=c1_channels, hid_dim=hid_dim)
        
        if self.aux:
            self.auxlayer = FCNHead(in_channels=728, channels=728 // 4, num_classes=self.nclass)

    def forward(self, x):
        size = x.shape[2:]
        c1, c2, c3, c4 = self.cnn_encoder(x)
        outputs = list()
        n, _, h, w = c4.shape
        c4 = self.hybrid_embed(c4)
        cls_token, c4 = self.transformer_encoder.forward_encoder(c4)
        attns_list = self.transformer_decoder.forward_decoder(c4)
        feat_enc = c4.reshape([n, h, w, -1]).transpose([0, 3, 1, 2])
        
        attn_map = attns_list[-1]
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape([B*nclass, nhead, H, W])
        x = paddle.concat([expand(feat_enc, nclass), attn_map], 1)
        x = self.cnn_head(x, c1, nclass, B)
        
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)
