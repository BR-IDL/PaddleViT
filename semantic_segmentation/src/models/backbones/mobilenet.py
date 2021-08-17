"""MobileNetV2."""
import paddle
import paddle.nn as nn

from .build import BACKBONE_REGISTRY
from ...modules import _ConvBNReLU, _DepthwiseConv, InvertedResidual

__all__ = ['MobileNetV2']


class MobileNetV2(nn.Layer):
    def __init__(self, config, num_classes=1000, norm_layer=nn.BatchNorm2D):
        super(MobileNetV2, self).__init__()
        output_stride = config.MODEL.OUTPUT_STRIDE
        self.multiplier = config.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
        elif output_stride == 16:
            dilations = [1, 2]
        elif output_stride == 8:
            dilations = [2, 4]
        else:
            raise NotImplementedError
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        # building first layer
        input_channels = int(32 * self.multiplier) if self.multiplier > 1.0 else 32
        # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
        self.conv1 = _ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)

        # building inverted residual blocks
        self.planes = input_channels
        self.block1 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[0:1],
                                       norm_layer=norm_layer)
        self.block2 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[1:2],
                                       norm_layer=norm_layer)
        self.block3 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[2:3],
                                       norm_layer=norm_layer)
        self.block4 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[3:5],
                                       dilations[0], norm_layer=norm_layer)
        self.block5 = self._make_layer(InvertedResidual, self.planes, inverted_residual_setting[5:],
                                       dilations[1], norm_layer=norm_layer)
        self.last_inp_channels = self.planes

        # weight initialization
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.KaimingNormal())
                if m.bias is not None:
                    m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=0.0))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=0.0))
            elif isinstance(m, nn.Linear):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.Normal(mean=0.0, std=0.01))
                if m.bias is not None:
                    m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=0.0))

    def _make_layer(self, block, planes, inverted_residual_setting, dilation=1, norm_layer=nn.BatchNorm2D):
        features = list()
        for t, c, n, s in inverted_residual_setting:
            out_channels = int(c * self.multiplier)
            stride = s if dilation == 1 else 1
            features.append(block(planes, out_channels, stride, t, dilation, norm_layer))
            planes = out_channels
            for i in range(n - 1):
                features.append(block(planes, out_channels, 1, t, norm_layer=norm_layer))
                planes = out_channels
        self.planes = planes
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        c1 = self.block2(x)
        c2 = self.block3(c1)
        c3 = self.block4(c2)
        c4 = self.block5(c3)

        return c1, c2, c3, c4


@BACKBONE_REGISTRY.register()
def mobilenet_v2(norm_layer=nn.BatchNorm2D):
    return MobileNetV2(norm_layer=norm_layer)

