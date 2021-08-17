"""Basic Module for Semantic Segmentation"""
import paddle
import paddle.nn as nn


__all__ = ['_ConvBNPReLU', '_ConvBN', '_BNPReLU', '_ConvBNReLU', '_DepthwiseConv', 'InvertedResidual',
           'SeparableConv2d']

_USE_FIXED_PAD = False


def _paddle_padding(kernel_size, stride=1, dilation=1, **_):
    if _USE_FIXED_PAD:
        return 0  # FIXME remove once verified
    else:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        # FIXME remove once verified
        fp = _fixed_padding(kernel_size, dilation)
        assert all(padding == p for p in fp)

        return padding


def _fixed_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return [pad_beg, pad_end, pad_beg, pad_end]


class SeparableConv2d(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2D):
        super().__init__()
        depthwise = nn.Conv2D(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias_attr=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2D(inplanes, planes, 1, bias_attr=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    )
        else:
            self.block = nn.Sequential(('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU()),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU())
                                                    )

    def forward(self, x):
        return self.block(x)


class _ConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2D):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ConvBNPReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2D):
        super(_ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=False)
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class _ConvBN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2D, **kwargs):
        super(_ConvBN, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_attr=False)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _BNPReLU(nn.Layer):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2D):
        super(_BNPReLU, self).__init__()
        self.bn = norm_layer(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        return x


# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
class _DepthwiseConv(nn.Layer):
    """conv_dw in MobileNet"""

    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.BatchNorm2D, **kwargs):
        super(_DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels, 3, stride, 1, groups=in_channels, norm_layer=norm_layer),
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(nn.Layer):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_layer=nn.BatchNorm2D):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2D(inter_channels, out_channels, 1, bias_attr=False),
            norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == '__main__':
    x = paddle.randn([1, 32, 64, 64])
    model = InvertedResidual(32, 64, 2, 1)
    out = model(x)
