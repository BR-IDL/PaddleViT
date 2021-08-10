import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PyramidPoolingModule(nn.Layer):
    """PyramidPoolingModule

    VisionTransformerUpHead is the decoder of PSPNet, Ref https://arxiv.org/abs/1612.01105.pdf

    Reference:                                                                                                                                                
        Hengshuang Zhao, et al. *"Pyramid Scene Parsing Network"*
    """
    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PyramidPoolingModule, self).__init__()   
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        norm_bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)) 

        self.pool_branches = nn.LayerList()
        for idx in range(len(self.pool_scales)):
            self.pool_branches.append( nn.Sequential(
                nn.AdaptiveAvgPool2D(self.pool_scales[idx]),
                nn.Conv2D(self.in_channels, self.channels, 1, stride=1, bias_attr=False),
                nn.SyncBatchNorm(self.channels, weight_attr=self.get_norm_weight_attr(), bias_attr=norm_bias_attr),
                nn.ReLU())) 

    def get_norm_weight_attr(self):
        return paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, x):
        outs = []
        up_resolution = [item for item in x.shape[2:]]
        for _, pool_layer in enumerate(self.pool_branches):
            out = pool_layer(x)
            up_out = F.interpolate(out, up_resolution, mode='bilinear', align_corners=self.align_corners)
            outs.append(up_out)
        return outs
            

            
