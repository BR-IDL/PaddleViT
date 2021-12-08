import numpy as np
import copy
import paddle
import paddle.nn as nn

"""
RegNet y-160
This is a simple version of regnet which only implements RegNetY-160.
This model is used as the teacher model for DeiT.
"""

class Identity(nn.Layer):
    """ Identity Layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SE(nn.Layer):
    """ Squeeze and Excitation module"""
    def __init__(self, in_channels, rd_channels, se_ratio=.25):
        super().__init__()
        if rd_channels is None:
            out_channels = int(in_channels * se_ratio)
        else:
            out_channels = rd_channels
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1_1x1 = nn.Conv2D(in_channels, out_channels, kernel_size=1)
        self.conv2_1x1 = nn.Conv2D(out_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv1_1x1(out)
        out = self.relu(out)
        out = self.conv2_1x1(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class Downsample(nn.Layer):
    """Downsample for 1st bottleneck block in every layer in RegNet"""
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1x1 = nn.Conv2D(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=stride,
                                 bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.bn(out)
        return out


class Bottleneck(nn.Layer):
    """Bottleneck residual block in Stage"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_ratio=1,
                 group_width=1,
                 stride=1,
                 dilation=1,
                 se_ratio=0.25):
        super().__init__()
        # 1x1 bottleneck conv block
        bottleneck_channels = int(round(out_channels * bottleneck_ratio))
        self.conv1 = nn.Conv2D(in_channels, bottleneck_channels, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(bottleneck_channels)
        # 3x3 conv block with group conv
        groups = bottleneck_channels // group_width
        self.conv2 = nn.Conv2D(bottleneck_channels,
                               bottleneck_channels,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=1,
                               groups=groups,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(bottleneck_channels)
        # SE modual
        if se_ratio:
            self.se = SE(bottleneck_channels, rd_channels=int(round(in_channels * se_ratio)))
        else:
            se_ratio = Identity()
        # downsample if stride = 2
        if stride != 1 or in_channels != out_channels:
            self.downsample = Downsample(in_channels, out_channels, stride)
        else:
            self.downsample = Identity()
        # 1x1 conv block
        self.conv3 = nn.Conv2D(bottleneck_channels,
                               out_channels,
                               kernel_size=1)
        self.bn3 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        h = self.downsample(h)

        out = out + h
        out = self.relu(out)
        return out


class RegStage(nn.Layer):
    """ Sequence of blocks with the same output shape"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 depth,
                 bottleneck_ratio,
                 group_width,
                 se_ratio=0.25):
        super().__init__()

        self.blocks = nn.LayerList()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                copy.deepcopy(Bottleneck(block_in_channels,
                                         out_channels,
                                         bottleneck_ratio,
                                         group_width,
                                         block_stride,
                                         se_ratio=se_ratio)))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RegNet(nn.Layer):
    """RegNet Model"""
    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg['num_classes']
        stem_width = cfg['stem_width']

        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv2D(in_channels=3,
                      out_channels=stem_width,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias_attr=False),
            nn.BatchNorm2D(stem_width),
            nn.ReLU())
        # RegStages
        self.stages = nn.LayerList()
        prev_width = stem_width
        curr_stride = 2
        stage_params = self._get_stage_params(cfg)
        for i, stage_param in enumerate(stage_params):
            self.stages.append(
                copy.deepcopy(RegStage(in_channels=prev_width,
                                       out_channels=stage_param['out_channels'],
                                       depth=stage_param['depth'],
                                       bottleneck_ratio=stage_param['bottle_ratio'],
                                       group_width=stage_param['group_width'],
                                       se_ratio=stage_param['se_ratio'])))
            prev_width = stage_param['out_channels']
        # Head
        num_features = prev_width
        self.head = nn.Sequential(nn.AdaptiveAvgPool2D(output_size=1),
                                  nn.Flatten(),
                                  nn.Linear(num_features, num_classes))

    def _get_stage_params(self, cfg):
        w_init = cfg['w0']
        w_slope = cfg['wa']
        w_mult = cfg['wm']
        depth = cfg['depth']
        se_ratio = cfg['se_ratio']
        group_w = cfg['group_w']
        bottle_ratio = cfg['bottle_ratio']

        w, d = self._generate_regnet(w_slope, w_init, w_mult, depth, bottle_ratio, group_w)

        num_stages = len(w)
        stage_widths = w
        stage_depths = d
        stage_bottle_ratios = [bottle_ratio for _ in range(num_stages)]
        stage_groups = [group_w for _ in range(num_stages)]
        se_ratios = [se_ratio for _ in range(num_stages)]
        param_names = ['out_channels', 'depth', 'bottle_ratio', 'group_width','se_ratio']
        stage_params = [
            dict(zip(param_names, params)) for params in zip(stage_widths,
                                                             stage_depths,
                                                             stage_bottle_ratios,
                                                             stage_groups,
                                                             se_ratios)]
        return stage_params 

    def _generate_regnet(self, w_slope, w_init, w_mult, depth, b=1, g=8):
        """Generate per block widths from RegNet parameters"""
        w_count = w_init + w_slope * np.arange(depth) # Equation 1
        w_exps = np.round(np.log(w_count / w_init) / np.log(w_mult)) # Equation 2
        
        w = w_init * np.power(w_mult, w_exps) # Equation 3
        w = np.round(np.divide(w, 8)) * 8 # make all width list divisible by 8

        w, d = np.unique(w.astype(int), return_counts=True) # find depth and width list

        gtemp = np.minimum(g, w//b)
        w = (np.round(w // b / gtemp) * gtemp).astype(int) # width

        return w, d

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


            
def build_regnet():
    """build regnet model using dict as config"""
    regnety_160 = {
        'stem_width': 32,
        'bottle_ratio': 1.0,
        'w0': 200,
        'wa': 106.23,
        'wm': 2.48,
        'group_w': 112,
        'depth': 18,
        'se_ratio': 0.25,
        'num_classes': 1000,
        'pool_size': (7, 7),
        'crop_pct': 0.875,
    }
    model = RegNet(regnety_160)
    return model
