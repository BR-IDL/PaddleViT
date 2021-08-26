import paddle
from paddle.fluid.layers.nn import size
import paddle.nn as nn
from src.models.backbones import VisualTransformer
from src.models.backbones import Deit
from src.models.decoders import MaskTransformer
from src.models.decoders import LinearDecoder

class Segmentor(nn.Layer):
    """
    Segmenter model implementation
    """
    def __init__(self, config):
        super().__init__()
        self.img_size = config.DATA.CROP_SIZE
        if 'ViT' in config.MODEL.ENCODER.TYPE:
            self.encoder = VisualTransformer(config)
        elif 'DeiT' in config.MODEL.ENCODER.TYPE:
            self.encoder = Deit(config)
        if 'MaskTransformer' in config.MODEL.DECODER_TYPE:
            self.decoder = MaskTransformer(config)
        elif 'Linear' in config.MODEL.DECODER_TYPE:
            self.decoder = LinearDecoder(config)
        self.norm = nn.LayerNorm(config.MODEL.TRANS.HIDDEN_SIZE)
        self.token_num = 2 if 'DeiT' in config.MODEL.ENCODER.TYPE else 1
        self.init__decoder_lr_coef(config.TRAIN.DECODER_LR_COEF)

    def init__decoder_lr_coef(self, coef):
        for param in self.decoder.parameters():
            param.optimize_attr['learning_rate'] = coef

    def forward(self, x):
        x = self.encoder(x)
        x = x[-1]
        x = self.norm(x)
        x = x[:, self.token_num:]
        masks = self.decoder(x)
        
        masks = nn.functional.interpolate(
            masks,
            size=self.img_size,
            mode="bilinear"
        )

        return [masks]