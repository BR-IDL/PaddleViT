import paddle.nn as nn

from .backbones.mix_transformer import MixVisionTransformer
from .decoders.segformer_head import SegformerHead


class Segformer(nn.Layer):
    """Segformer model implementation
    
    """
    def __init__(self, config):
        super(Segformer, self).__init__()
        self.backbone = MixVisionTransformer(
            in_channels=config.MODEL.TRANS.IN_CHANNELS,
            embed_dims=config.MODEL.TRANS.EMBED_DIM,
            num_stages=config.MODEL.TRANS.NUM_STAGES,
            num_layers=config.MODEL.TRANS.NUM_LAYERS,
            num_heads=config.MODEL.TRANS.NUM_HEADS,
            patch_sizes=config.MODEL.TRANS.PATCH_SIZE,
            strides=config.MODEL.TRANS.STRIDES,
            sr_ratios=config.MODEL.TRANS.SR_RATIOS,
            out_indices=config.MODEL.ENCODER.OUT_INDICES,
            mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
            qkv_bias=config.MODEL.TRANS.QKV_BIAS,
            drop_rate=config.MODEL.DROPOUT,
            attn_drop_rate=config.MODEL.ATTENTION_DROPOUT,
            drop_path_rate=config.MODEL.DROP_PATH,
            pretrained=config.MODEL.PRETRAINED)
        self.decode_head = SegformerHead(
            in_channels=config.MODEL.SEGFORMER.IN_CHANNELS,
            channels=config.MODEL.SEGFORMER.CHANNELS,
            num_classes=config.DATA.NUM_CLASSES,
            align_corners=config.MODEL.SEGFORMER.ALIGN_CORNERS)

    def forward(self, inputs):
        features = self.backbone(inputs)
        out = self.decode_head(features)
        return out