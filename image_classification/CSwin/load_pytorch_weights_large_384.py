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
from cswin import *
from cswin_pytorch.CSWin_Transformer.models import cswin as pytorch_cswin
from config import *

config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg', type=str, default='./configs/cswin_large_384.yaml')
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-image_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-eval', action="store_true")
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
args = parser.parse_args()

config = get_config()
config = update_config(config, args)
print(config)


def print_model_named_params(model):
    print('-----------------MODEL NAMED PARAMETERS------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)


def print_model_named_buffers(model):
    print('-----------------MODEL NAMED BUFFERS------------------------')
    for name, buff in model.named_buffers():
        print(name, buff.shape)


def torch_to_paddle_mapping():
    mapping = [
        ('stage1_conv_embed.0', 'patch_embedding.patch_embed'),
        ('stage1_conv_embed.2', 'patch_embedding.norm'),
    ]

    for stage_idx, stage_depth in enumerate(config.MODEL.TRANS.DEPTHS):
        for idx in range(stage_depth):
            th_prefix = f'stage{stage_idx+1}.{idx}'
            pp_prefix = f'stages.{stage_idx}.blocks.{idx}'

            layer_mapping = [
                (f'{th_prefix}.qkv', f'{pp_prefix}.qkv'),
                (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                (f'{th_prefix}.proj', f'{pp_prefix}.proj'),
                (f'{th_prefix}.attns.0.get_v', f'{pp_prefix}.attns.0.get_v'),
                (f'{th_prefix}.attns.1.get_v', f'{pp_prefix}.attns.1.get_v'), # may not exist, ok
                (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'), 
                (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'), 
            ]
            mapping.extend(layer_mapping)
        # prefix for last stage may not exist, it is ok in this code
        th_prefix = f'merge{stage_idx+1}'
        pp_prefix = f'stages.{stage_idx}.merge'
        layer_mapping = [
            (f'{th_prefix}.conv', f'{pp_prefix}.conv'),
            (f'{th_prefix}.norm', f'{pp_prefix}.norm'),
        ]
        mapping.extend(layer_mapping)

    head_mapping = [
        ('norm', 'norm'),
        ('head', 'head'),
    ]
    mapping.extend(head_mapping)

    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
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
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():

    paddle.set_device('cpu')
    paddle_model = build_cswin(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('----------------------------------')

    device = torch.device('cpu')
    torch_model = pytorch_cswin.CSWin_144_24322_large_384(img_size=384)
    model_state_dict = torch.load('./cswin_pytorch/cswin_large_384.pth')
    torch_model.load_state_dict(model_state_dict['state_dict_ema'])
    torch_model = torch_model.to(device)
    torch_model.eval()

    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, 384, 384).astype('float32')
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
    model_path = os.path.join('./cswin_large_384.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
