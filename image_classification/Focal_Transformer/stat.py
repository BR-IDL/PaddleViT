import os
import glob
import paddle
from config import get_config
from swin_transformer import build_swin as build_model

def count_gelu(layer, input_, output):
    activation_flops = 8
    x = input_[0]
    num = x.numel()
    layer.total_ops += num * activation_flops 


def count_softmax(layer, input_, output):
    softmax_flops = 5 # max/substract, exp, sum, divide
    x = input_[0]
    num = x.numel()
    layer.total_ops += num * softmax_flops 


def count_layernorm(layer, input_, output):
    layer_norm_flops = 5 # get mean (sum), get variance (square and sum), scale(multiply)
    x = input_[0]
    num = x.numel()
    layer.total_ops += num * layer_norm_flops 


cfg = './configs/swin_tiny_patch4_window7_224.yaml'
input_size = (1, 3, 224, 224)
config = get_config(cfg)
model = build_model(config)

custom_ops = {paddle.nn.GELU: count_gelu,
              paddle.nn.LayerNorm: count_layernorm,
              paddle.nn.Softmax: count_softmax,
            }
print(os.path.basename(cfg))
paddle.flops(model,
             input_size=input_size,
             custom_ops=custom_ops,
             print_detail=False)
