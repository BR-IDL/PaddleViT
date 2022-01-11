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
import timm
from hvt import build_hvt
from model_th import hvt_model
import os
from config import *
import json

model_name = 'scale_hvt_ti_4_patch16_224'
sz = int(model_name[-3::])

config = get_config(f'./configs/{model_name}.yaml')


def print_model_named_params(model):
    print('----------------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)
    print('----------------------------------')


def print_model_named_buffers(model):
    print('----------------------------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
    print('----------------------------------')


def torch_to_paddle_mapping():
    mapping = [
        ('pos_embed', 'pos_embed'),
        ('patch_embed.proj', f'patch_embed.proj'),
        ('blocks.0.pos_embed', f'layers.0.pos_embed'),
        ('blocks.3.pos_embed', f'layers.3.pos_embed'),
        ('blocks.6.pos_embed', f'layers.6.pos_embed'),
        ('blocks.9.pos_embed', f'layers.9.pos_embed')
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
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2')
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
    paddle.set_device('cpu')
    paddle_model = build_hvt(config)
    paddle_model.eval()

    # print(paddle_model)
    # print_model_named_params(paddle_model)
    # print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    torch_model = hvt_model()
    # print(torch_model)
    # print_model_named_params(torch_model)
    # print_model_named_buffers(torch_model)

    checkpoint = torch.load('scale_hvt_ti_4.pth')['model']
    for sub_item in checkpoint:
        print(sub_item)

    torch_model.load_state_dict(checkpoint)
    torch_model = torch_model.to(device)
    torch_model.eval()


    print('+++++++++++++++++++++++++++++++++++')

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, sz, sz).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_paddle = paddle_model(x_paddle)
    out_paddle = out_paddle.cpu().numpy()

    out_torch = torch_model(x_torch)
    out_torch = out_torch.detach().cpu().numpy()

    # for out_paddle,out_torch in zip(out_paddle_np,out_torch_np):
    out_diff = np.allclose(out_torch, out_paddle, atol=1e-5)
    print(out_diff)
    print(np.sum(out_torch), np.sum(out_paddle))

    assert np.allclose(out_torch, out_paddle, atol=1e-5)

    # save weights for paddle model
    model_path = os.path.join(f'./{model_name}.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)
    print('all done')


if __name__ == "__main__":
    main()
