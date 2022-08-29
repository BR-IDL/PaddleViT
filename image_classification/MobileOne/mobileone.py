import copy
import numpy as np
import paddle
import paddle.nn as nn


class ConvNormAct(nn.Sequential):
    """Layer ops: Conv2D -> NormLayer -> ActLayer"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D):
        layers = [('conv', nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
                              bias_attr=bias_attr))]
        if norm_layer is not None:
            layers.append(('norm', norm_layer(out_channels)))
        if act_layer is not None:
            layers.append(('act', act_layer()))

        super().__init__(*layers)


class SEBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 rd_ratio=0.0625):
        super().__init__()
        self.reduce = nn.Conv2D(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1)
        self.expand = nn.Conv2D(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.reduce(x)
        x = self.relu(x)
        x = self.expand(x)
        x = self.sigmoid(x)
        return inputs * x


class MobileOneBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_branches,
                 stride=1,
                 use_se=False,
                 deploy=False,
                 use_pw=True):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = num_branches
        self.use_pw = use_pw
        self.act = nn.ReLU()
        if use_se:
            self.dw_se = SEBlock(in_channels)
            self.pw_se = SEBlock(out_channels) if use_pw else None
        else:
            self.dw_se = nn.Identity()
            self.pw_se = nn.Identity()

        if deploy:
            self.dw_3x3 = ConvNormAct(in_channels=in_channels,
                                      out_channels=in_channels if use_pw else out_channels,
                                      kernel_size=3,
                                      padding=1,
                                      stride=stride,
                                      groups=in_channels if use_pw else 1,
                                      bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
                                      norm_layer=None,
                                      act_layer=None)
            self.pw_1x1 = ConvNormAct(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      groups=1,
                                      bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
                                      norm_layer=None,
                                      act_layer=None) if use_pw else None
        else:
            # in mobileone stride == 1 is equal to in_channels == out_channels
            self.dw_skip = nn.BatchNorm2D(in_channels) if stride == 1 else None  # rbr_skip
            self.dw_1x1 = ConvNormAct(in_channels=in_channels,  # rbr_scale
                                      out_channels=in_channels if use_pw else out_channels,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      groups=in_channels if use_pw else 1,
                                      bias_attr=False,
                                      act_layer=None,
                                      norm_layer=nn.BatchNorm2D)
            self.dw_3x3_blocks = nn.LayerList([  # rbr_conv
                ConvNormAct(in_channels=in_channels,
                            out_channels=in_channels if use_pw else out_channels,
                            kernel_size=3,
                            stride=stride,
                            padding=1,
                            groups=in_channels if use_pw else 1,
                            bias_attr=False,
                            act_layer=None,
                            norm_layer=nn.BatchNorm2D) for i in range(num_branches)])
            self.pw_skip = nn.BatchNorm2D(in_channels) if in_channels == out_channels and use_pw else None  # rbr_skip
            self.pw_1x1_blocks = nn.LayerList([  # rbr_conv
                ConvNormAct(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            bias_attr=False,
                            act_layer=None,
                            norm_layer=nn.BatchNorm2D) for i in range(num_branches)]) if use_pw else None

    def forward(self, x):
        if self.deploy:
            print(x.shape)
            print(self.dw_3x3)
            out = self.dw_3x3(x)
            out = self.dw_se(out)
            out = self.act(out)
            if self.use_pw:
                out = self.pw_1x1(out)
                out = self.pw_se(out)
                out = self.act(out)
        else:
            dw_out = self.dw_1x1(x)
            #print('dw_1x1 out: ', dw_out[0,0,0,:5])
            dw_out +=  sum([dw_3x3(x) for dw_3x3 in self.dw_3x3_blocks])
            #dw_out = self.dw_1x1(x) + sum([dw_3x3(x) for dw_3x3 in self.dw_3x3_blocks])
            dw_out += self.dw_skip(x) if self.dw_skip is not None else 0
            #print('dw_out: ', dw_out[0,0,0,:5])
            dw_out = self.dw_se(dw_out)
            dw_out = self.act(dw_out)
    
            if self.use_pw:
                out = sum([pw_1x1(dw_out) for pw_1x1 in self.pw_1x1_blocks])
                #print('pw_out_before_skip: ', out[0,0,0,:5])
                #out += self.pw_skip(dw_out) if self.pw_skip is not None else 0
                skip_out = self.pw_skip(dw_out) if self.pw_skip is not None else 0
                #if self.pw_skip:
                #    print('skip_out: ', skip_out[0,0,0,:5])
                out = out + skip_out
                #print('pw_out: ', out[0,0,0,:5])
                out = self.pw_se(out)
                out = self.act(out)
            else:
                out = dw_out
        return out
        
    def get_equivalent_kernel_bias(self):
        dw_kernel_3x3 = []
        dw_bias_3x3 = []
        for k_idx in range(self.num_branches):
            k3, b3 = self._fuse_bn_tensor(self.dw_3x3_blocks[k_idx])
            dw_kernel_3x3.append(k3)
            dw_bias_3x3.append(b3)

        dw_kernel_1x1, dw_bias_1x1 = self._fuse_bn_tensor(self.dw_1x1)
        dw_kernel_identity, dw_bias_identity = self._fuse_bn_tensor(self.dw_skip, self.in_channels)
        dw_kernel = sum(dw_kernel_3x3) + self._pad_1x1_to_3x3_tensor(dw_kernel_1x1) + dw_kernel_identity
        dw_bias = sum(dw_bias_3x3) + dw_bias_1x1 + dw_bias_identity


        if self.use_pw:
            pw_kernel = []
            pw_bias = []
            for k_idx in range(self.num_branches):
                k1, b1 = self._fuse_bn_tensor(self.pw_1x1_blocks[k_idx])
                pw_kernel.append(k1)
                pw_bias.append(b1)
            pw_kernel_identity, pw_bias_identity = self._fuse_bn_tensor(self.pw_skip, 1)
            pw_kernel_1x1 = sum(pw_kernel) + pw_kernel_identity
            pw_bias_1x1 = sum(pw_bias) + pw_bias_identity
        else:
            pw_kernel_1x1 = None
            pw_bias_1x1 = None

        return dw_kernel, dw_bias, pw_kernel_1x1, pw_bias_1x1

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            return paddle.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch, groups=None):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            assert bias is None
            running_mean = branch.norm._mean
            running_var = branch.norm._variance
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            input_dim = self.in_channels // groups
            kernel_size = 1 if groups == 1 else 3
            kernel_value = np.zeros((self.in_channels, input_dim, kernel_size, kernel_size),
                                    dtype=np.float32)
            for i in range(self.in_channels):
                if kernel_size == 1:
                    kernel_value[i, i % input_dim, 0, 0] = 1
                else:
                    kernel_value[i, i % input_dim, 1, 1] = 1 

            self.identity_tensor = paddle.to_tensor(kernel_value, dtype='float32')
            kernel = self.identity_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))

        return kernel * t,  -running_mean * gamma / std + beta

    def switch_to_deploy(self):
        dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        self.dw_3x3 = ConvNormAct(
            in_channels=self.dw_3x3_blocks[0].conv._in_channels,
            out_channels=self.dw_3x3_blocks[0].conv._in_channels if self.use_pw else self.dw_3x3_blocks[0].conv._out_channels,
            kernel_size=self.dw_3x3_blocks[0].conv._kernel_size,
            stride=self.dw_3x3_blocks[0].conv._stride,
            padding=self.dw_3x3_blocks[0].conv._padding,
            groups=self.dw_3x3_blocks[0].conv._in_channels if self.use_pw else 1,
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
            norm_layer=None,
            act_layer=nn.ReLU)
        self.dw_3x3.conv.weight.set_value(dw_kernel)
        self.dw_3x3.conv.bias.set_value(dw_bias)

        if self.use_pw:
            self.pw_1x1 = ConvNormAct(
                in_channels=self.pw_1x1_blocks[0].conv._in_channels,
                out_channels=self.pw_1x1_blocks[0].conv._out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.0)),
                norm_layer=None,
                act_layer=nn.ReLU)
            self.pw_1x1.conv.weight.set_value(pw_kernel)
            self.pw_1x1.conv.bias.set_value(pw_bias)

        for param in self.parameters():
            param.detach()

        self.__delattr__('dw_1x1')
        self.__delattr__('dw_3x3_blocks')
        if hasattr(self, 'dw_skip'):
            self.__delattr__('dw_skip')

        if self.use_pw:
            self.__delattr__('pw_1x1_blocks')
            if hasattr(self, 'pw_skip'):
                self.__delattr__('pw_skip')
            if hasattr(self, 'identity_tensor'):
                self.__delattr__('identity_tensor')

        self.deploy = True


class MobileOne(nn.Layer):
    def __init__(self,
                 num_blocks,
                 num_branches,
                 channels,
                 strides,
                 expansions,
                 num_classes=1000,
                 use_se=False,
                 deploy=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_branches = num_branches
        self.channels = channels
        self.strides = strides
        self.expansions = expansions
        self.num_classes = num_classes
        self.use_se = use_se
        self.deploy = deploy
        self.num_stages = len(num_blocks)
        self.stages = []
        in_channels = 3

        # create stages
        for stage_idx, (nb, br, ch, s, e) in enumerate(zip(num_blocks, num_branches, channels, strides, expansions)):
            stage_blocks = []
            # each stage contains a number of blocks
            for block_idx in range(nb):
                # set up se according to official github code
                if stage_idx <= 2:  # no se used
                    block_use_se = False
                elif stage_idx == 3: # use se on half blocks
                    if block_idx < nb // 2:
                        block_use_se = False
                    else:
                        block_use_se = use_se
                elif stage_idx == 4:
                    block_use_se = use_se
                    
                out_channels = int(ch * e)
                stage_blocks.append(MobileOneBlock(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   num_branches=1 if stage_idx==0 else br,
                                                   stride=s if block_idx == 0 else 1,
                                                   use_se=block_use_se,
                                                   deploy=deploy,
                                                   use_pw=False if stage_idx==0 else True))
                in_channels = out_channels
            self.stages.append(nn.Sequential(*stage_blocks))
        self.stages = nn.LayerList(self.stages)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(out_channels, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        for idx, stage in enumerate(self.stages):
            #if idx == 1:
            #    print('************ stage 1')
            x = stage(x)
            #if (idx == 2):
            #    print('********* paddle stage2: ', x)
        x = self.avg_pool(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


def model_convert(model, inputs=None):
    model_convert = copy.deepcopy(model)
    model_convert.eval()
    for layer in model_convert.sublayers():
        if hasattr(layer, 'switch_to_deploy'):
            layer.switch_to_deploy()

    deploy_model = MobileOne(num_blocks=model.num_blocks,
                             num_branches=model.num_branches,
                             channels=model.channels,
                             strides=model.strides,
                             expansions=model.expansions,
                             num_classes=model.num_classes,
                             use_se=model.use_se,
                             deploy=True) 
    deploy_model.eval()
    model.eval()

    #print('model =====================')
    #for name, val in model_convert.state_dict().items():
    #    print(name, val.shape)
    #print('deploy_model =====================')
    #for name, val in deploy_model.state_dict().items():
    #    print(name, val.shape)

    deploy_model.set_state_dict(model_convert.state_dict())
    # check
    if inputs is not None:
        out_deploy = deploy_model(inputs)
        out_train = model(inputs)
        #print(deploy_model)
        print(out_train)
        print('===================')
        print(out_deploy)
        print('========== deploy diff ==============')
        print((out_train - out_deploy))

    return deploy_model


def build_mobileone(config):
    """Build MobileOne by reading options in config object
    Args:
        config: config instance contains setting options
    Returns:
        model: MobileOne model
    """
    model = MobileOne(num_blocks=config.MODEL.NUM_BLOCKS,
                      num_branches=config.MODEL.NUM_BRANCHES,
                      channels=config.MODEL.CHANNELS,
                      strides=config.MODEL.STRIDES,
                      expansions=config.MODEL.EXPANSIONS,
                      num_classes=config.MODEL.NUM_CLASSES,
                      use_se=config.MODEL.USE_SE,
                      deploy=config.MODEL.DEPLOY)
    return model






