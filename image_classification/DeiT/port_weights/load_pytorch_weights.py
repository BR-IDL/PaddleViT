#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

import argparse
import numpy as np
import paddle
import torch
from deit import *
from config import *
from stats import count_gelu, count_softmax, count_layernorm


model_name = 'deit_tiny_distilled_patch16_224'
cfg_name = 'deit_tiny_patch16_224'
sz = 224

config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg', type=str, default=f'./configs/{cfg_name}.yaml')
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-image_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-eval', action="store_true")
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-teacher_model', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
args = parser.parse_args()

config = get_config()
config = update_config(config, args)
print(config)


def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)

def torch_to_paddle_mapping():
    mapping = [
        ('cls_token', 'class_token'),
        ('dist_token', 'distill_token'),
        ('pos_embed', 'pos_embed'),
        ('patch_embed.proj', f'patch_embed.proj'),
    ]

    num_layers = config.MODEL.TRANS.DEPTH
    for idx in range(num_layers):
        th_prefix = f'blocks.{idx}'
        pp_prefix = f'layers.{idx}'
        layer_mapping = [
            (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
            (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
            (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
            (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
            (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'), 
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'), 
        ]
        mapping.extend(layer_mapping)

    head_mapping = [
        ('norm', 'norm'),
        ('head', 'head'),
        ('head_dist', 'head_distill')
    ]
    mapping.extend(head_mapping)

    return mapping



def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].cpu().data.numpy()
        if len(value.shape) == 2:
            value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)

    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in paddle_model.named_buffers():
        pd_params[name] = param

    for name, param in torch_model.named_parameters():
        th_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()
    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            _set_value(th_name, pd_name)
        else: # weight & bias
            th_name_w = f'{th_name}.weight'
            pd_name_w = f'{pd_name}.weight'
            _set_value(th_name_w, pd_name_w)

            th_name_b = f'{th_name}.bias'
            pd_name_b = f'{pd_name}.bias'
            _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():

    #paddle.set_device('cpu')
    paddle_model = build_deit(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print('--------------')
    print_model_named_buffers(paddle_model)
    print('----------------------------------')

    device = torch.device('cuda')
    torch_model = torch.hub.load('facebookresearch/deit:main',
                                 f'{model_name}', #'deit_base_distilled_patch16_224',
                                 pretrained=True)
    torch_model = torch_model.to(device)
    torch_model.eval()

    print_model_named_params(torch_model)
    print('--------------')
    print_model_named_buffers(torch_model)
    print('----------------------------------')


    #return

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, sz, sz).astype('float32')
    #x = np.random.randn(2, 3, 224, 224).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torch_model(x_torch)
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:100])
    print(out_paddle[0, 0:100])
    assert np.allclose(out_torch, out_paddle, atol = 1e-5)
    
    # save weights for paddle model
    #model_path = os.path.join('./deit_base_distilled_patch16_224.pdparams')
    model_path = os.path.join(f'./{model_name}.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)

    custom_ops = {paddle.nn.GELU: count_gelu,
                    paddle.nn.LayerNorm: count_layernorm,
                    paddle.nn.Softmax: count_softmax,
    }
    paddle.flops(paddle_model,
                 input_size=(1, 3, sz, sz),
                 custom_ops=custom_ops,
                 print_detail=False)

if __name__ == "__main__":
    main()
