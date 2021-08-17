import paddle.nn as nn
import paddle

from .build import BACKBONE_REGISTRY

__all__ = ['ResNetV1']


class BasicBlockV1b(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2D(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(nn.Layer):

    def __init__(self, block, layers, config, num_classes=1000, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2D):
        output_stride = config.MODEL.OUTPUT_STRIDE
        scale = config.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(ResNetV1, self).__init__()
        if deep_stem:
            # resnet vc
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, mid_channel, 3, 2, 1, bias_attr=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2D(mid_channel, mid_channel, 3, 1, 1, bias_attr=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2D(mid_channel, self.inplanes, 3, 1, 1, bias_attr=False)
            )
        else:
            self.conv1 = nn.Conv2D(3, self.inplanes, 7, 2, 3, bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2D(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2, norm_layer=norm_layer)

        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer, multi_grid=config.MODEL.DANET.MULTI_GRID,
                                       multi_dilation=config.MODEL.DANET.MULTI_DILATION)

        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight = paddle.create_parameter(shape=m.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(shape=m.bias.shape, dtype='float32', default_initializer=nn.initializer.Constant(value=0.0))

        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, BottleneckV1b):
                    m.bn3.weight = paddle.create_parameter(shape=m.bn3.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(0.0))
                elif isinstance(m, BasicBlockV1b):
                    m.bn2.weight = paddle.create_parameter(shape=m.bn2.weight.shape, dtype='float32', default_initializer=nn.initializer.Constant(0.0))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2D,
                    multi_grid=False, multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride, bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div],
                                    previous_dilation=dilation, norm_layer=norm_layer))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation,
                                    previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # for classification
        # x = self.avgpool(c4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return c1, c2, c3, c4


@BACKBONE_REGISTRY.register()
def resnet50(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 4, 6, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet101(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 4, 23, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet152(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 8, 36, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet50c(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 4, 6, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer, deep_stem=True)


@BACKBONE_REGISTRY.register()
def resnet101c(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 4, 23, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer, deep_stem=True)


@BACKBONE_REGISTRY.register()
def resnet152c(config, norm_layer=nn.BatchNorm2D):
    num_block = [3, 8, 36, 3]
    return ResNetV1(BottleneckV1b, num_block, config, norm_layer=norm_layer, deep_stem=True)

