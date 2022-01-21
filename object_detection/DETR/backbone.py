# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

"""
Backbone related classes and methods for DETR
Backbone now supports ResNet50, and ResNet101
'build_backbone' method returns resnet with position_embedding
ResNet is implemented in ./resnet.py
"""

from collections import OrderedDict
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from position_embedding import build_position_encoding
import resnet
from utils import NestedTensor


class IntermediateLayerGetter(nn.LayerDict):
    """ Run inference and return outputs from selected layers

    This class stores the layers needed for inferening all the selected
    return layers, layers after the last return layer will be ignored.
    Forward method returns a dict with layer names and corresponding output tensors

    Arguments:
        model: nn.Layer, backbone model, e.g., resnet50
        return_layers:  dict, dict of return layers
    """

    def __init__(self, model, return_layers):
        #print([name for name, _ in model.named_children()])
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        # copy return_layers is required, otherwise orig_return_layers will be empty
        return_layers = {k:v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FrozenBatchNorm2D(nn.Layer):
    """Freeze the bn layer without learning and updating.

    This layer can be replaced with nn.BatchNorm2D, in order to freeze
    the learning, usually is used in backbone during the training.
    Weights and params are same as nn.BatchNorm2D, now eps is set to 1e-5
    """

    def __init__(self, n):
        super(FrozenBatchNorm2D, self).__init__()
        self.register_buffer('weight', paddle.ones([n]))
        self.register_buffer('bias', paddle.zeros([n]))
        self.register_buffer('_mean', paddle.zeros([n]))
        self.register_buffer('_variance', paddle.ones([n]))

    def forward(self, x):
        w = self.weight.reshape([1, -1, 1, 1])
        b = self.bias.reshape([1, -1, 1, 1])
        rv = self._variance.reshape([1, -1, 1, 1])
        rm = self._mean.reshape([1, -1, 1, 1])
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Layer):
    """Backbone Base class for NestedTensor input and multiple outputs

    This class handles the NestedTensor as input, run inference through backbone
    and return multiple output tensors(NestedTensors) from selected layers
    """

    def __init__(self,
                 backbone: nn.Layer,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.stop_gradient = True

        if return_interm_layers:
            return_layers = {'layer0': '0', 'layer2': '1', 'layer3':'2', 'layer4':'3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        #Inference through resnet backbone, which takes the paddle.Tensor as input
        #tensor_list contains .tensor(paddle.Tensor) and .mask(paddle.Tensor) for batch inputs
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            # x.shape: [batch_size, feat_dim, feat_h, feat_w]
            m = tensor_list.mask  # [batch_size, orig_h, orig_w]
            assert m is not None
            m = m.unsqueeze(0).astype('float32') # [1, batch_size, orig_h, orig_w]
            mask = F.interpolate(m, size=x.shape[-2:])[0] #[batch_size, feat_h, fea_w]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """Get resnet backbone from resnet.py with multiple settings and return BackboneBase instance"""
    def __init__(self, name, train_backbone, return_interm_layers, dilation, backbone_lr):
        backbone = getattr(resnet, name)(pretrained=paddle.distributed.get_rank() == 0,
                                         norm_layer=FrozenBatchNorm2D,
                                         replace_stride_with_dilation=[False, False, dilation],
                                         backbone_lr=backbone_lr)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """ Joiner layers(nn.Sequential) for backbone and pos_embed
    Arguments:
        backbone: nn.Layer, backbone layer (resnet)
        position_embedding: nn.Layer, position_embedding(learned, or sine)
    """
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        # feature from resnet backbone inference
        xs = self[0](x)
        out = []
        pos = []
        # for each backbone output, apply position embedding
        for name, xx in xs.items():
            out.append(xx)
            pos.append(self[1](xx).astype(xx.tensors.dtype))
        return out, pos


def build_backbone(config):
    """ build resnet backbone and position embedding according to config """
    assert config.MODEL.BACKBONE in ['resnet50', 'resnet101'], "backbone name is not supported!"
    backbone_name = config.MODEL.BACKBONE
    dilation = False
    train_backbone = not config.EVAL
    return_interm_layers = False #TODO: impl case True for segmentation
    backbone_lr = config.MODEL.BACKBONE_LR

    position_embedding = build_position_encoding(config.MODEL.TRANS.EMBED_DIM)
    backbone = Backbone(backbone_name, train_backbone, return_interm_layers, dilation, backbone_lr)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model
