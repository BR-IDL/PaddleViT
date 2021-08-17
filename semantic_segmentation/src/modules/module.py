"""Basic Module for Semantic Segmentation"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from collections import OrderedDict
from .basic import _ConvBNReLU, SeparableConv2d, _ConvBN, _BNPReLU, _ConvBNPReLU

__all__ = ['_FCNHead', '_ASPP', 'PyramidPooling', 'PAM_Module', 'CAM_Module', 'EESP']


class _FCNHead(nn.Layer):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2D):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2D(in_channels, inter_channels, 3, padding=1, bias_attr=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2D(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------
#                      For deeplab
# -----------------------------------------------------------------
class _ASPP(nn.Layer):
    def __init__(self, config, in_channels=2048, out_channels=256):
        super().__init__()
        output_stride = config.MODEL.OUTPUT_STRIDE
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)),
                                                ('bn', nn.BatchNorm2D(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)),
                                                        ('bn', nn.BatchNorm2D(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2D(out_channels*5, out_channels, 1, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = paddle.concat((pool, x0, x1, x2, x3), axis=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

# -----------------------------------------------------------------
#                 For PSPNet, fast_scnn
# -----------------------------------------------------------------
class PyramidPooling(nn.Layer):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2D, **kwargs):
        super(PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpools = nn.ModuleList()
        self.convs = nn.ModuleList()
        for size in sizes:
            self.avgpools.append(nn.AdaptiveAvgPool2d(size))
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer, **kwargs))

    def forward(self, x):
        size = x.size()[2:]
        feats = [x]
        for (avgpool, conv) in zip(self.avgpools, self.convs):
            feats.append(F.interpolate(conv(avgpool(x)), size, mode='bilinear', align_corners=True))
        return paddle.concat(feats, axis=1)


class PAM_Module(nn.Layer):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(paddle.zeros(1))
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).reshape([m_batchsize, -1, width*height]).transpose([0, 2, 1])
        proj_key = self.key_conv(x).reshape([m_batchsize, -1, width*height])
        energy = paddle.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).reshape([m_batchsize, -1, width*height])

        out = paddle.bmm(proj_value, attention.transpose([0, 2, 1]))
        out = out.reshape([m_batchsize, C, height, width])

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Layer):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(paddle.zeros([1]))
        self.softmax  = nn.Softmax(axis=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.reshape([m_batchsize, C, -1])
        proj_key = x.reshape([m_batchsize, C, -1]).transpose(0, 2, 1)
        energy = paddle.bmm(proj_query, proj_key)
        energy_new = paddle.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.reshape([m_batchsize, C, -1])

        out = paddle.bmm(attention, proj_value)
        out = out.reshape([m_batchsize, C, height, width])

        out = self.gamma*out + x
        return out


class EESP(nn.Layer):

    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7, down_method='esp', norm_layer=nn.BatchNorm2D):
        super(EESP, self).__init__()
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = _ConvBNPReLU(in_channels, n, 1, stride=1, groups=k, norm_layer=norm_layer)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            dilation = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(nn.Conv2D(n, n, 3, stride, dilation, dilation=dilation, groups=n, bias_attr=False))
        self.conv_1x1_exp = _ConvBN(out_channels, out_channels, 1, 1, groups=k, norm_layer=norm_layer)
        self.br_after_cat = _BNPReLU(out_channels, norm_layer)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)
        expanded = self.conv_1x1_exp(self.br_after_cat(paddle.concat(output, 1)))
        del output
        if self.stride == 2 and self.downAvg:
            return expanded

        if expanded.size() == x.size():
            expanded = expanded + x

        return self.module_act(expanded)
