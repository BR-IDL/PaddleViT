from collections import OrderedDict
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from position_embedding import build_position_encoding
import resnet
from utils import NestedTensor



class IntermediateLayerGetter(nn.LayerDict):
    def __init__(self, model, return_layers):
        #print([name for name, _ in model.named_children()])

        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {k:v for k, v in return_layers.items()}
        layers =OrderedDict()
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
            #print(f'--------{name}-------------')
            #print(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FrozenBatchNorm2D(nn.Layer):
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
    def __init__(self,
                 backbone: nn.Layer,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool):
        super().__init__()
        #for name, param in backbone.named_parameters():
        #    if not train_backbone


        if return_interm_layers:
            return_layers = {'layer0': '0', 'layer2': '1', 'layer3':'2', 'layer4':'3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = {}
        for name, x in xs.items():
            # x.shape: [batch_size, feat_dim, feat_h, feat_w]
            m = tensor_list.mask  # [batch_size, orig_h, orig_w]
            #print(m[0])
            assert m is not None
            m = m.unsqueeze(0).astype('float32') # [1, batch_size, orig_h, orig_w]
            mask = F.interpolate(m, size=x.shape[-2:])[0] #[batch_size, feat_h, fea_w]
            #print(mask)
            out[name] = NestedTensor(x, mask)

        return out

class Backbone(BackboneBase):
    def __init__(self, name, train_backbone, return_interm_layers, dilation):
        backbone = getattr(resnet, name)(pretrained=paddle.distributed.get_rank()==0,
                                         norm_layer=FrozenBatchNorm2D,
                                         replace_stride_with_dilation=[False, False, dilation])
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        
    def forward(self, x):
        xs = self[0](x)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).astype(x.tensors.dtype))
            #print(f'----- {name} pos: ---------')
            #print(pos[-1])
        return out, pos



def build_backbone():
    backbone_name = 'resnet50'
    dilation = False
    train_backbone = True
    #return_interm_layers=True #TODO: add for segmentation
    return_interm_layers=False #TODO: add for segmentation

    position_embedding = build_position_encoding()
    backbone = Backbone(backbone_name, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

