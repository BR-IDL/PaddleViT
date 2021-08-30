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

import os
import argparse
import numpy as np
import paddle
import torch
import timm
from resmlp import build_res_mlp
from config import get_config
from config import update_config

config = get_config('./configs/resmlp_big_24_224.yaml')
print(config)


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
        ('stem.proj', 'patch_embed.patch_embed'),
        ('norm.alpha', 'norm.alpha'),
        ('norm.beta', 'norm.beta'),
        ('head', 'head'),
    ]

    num_layers = 24
    for idx in range(num_layers):
        pp_prefix = f'mixer_layers.{idx}'
        th_prefix = f'blocks.{idx}'
        layer_mapping = [
            (f'{th_prefix}.ls1', f'{pp_prefix}.ls1'),
            (f'{th_prefix}.ls2', f'{pp_prefix}.ls2'),
            (f'{th_prefix}.norm1.alpha', f'{pp_prefix}.norm1.alpha'),
            (f'{th_prefix}.norm1.beta', f'{pp_prefix}.norm1.beta'),
            (f'{th_prefix}.norm2.alpha', f'{pp_prefix}.norm2.alpha'),
            (f'{th_prefix}.norm2.beta', f'{pp_prefix}.norm2.beta'),
            (f'{th_prefix}.linear_tokens', f'{pp_prefix}.linear_tokens'),
            (f'{th_prefix}.mlp_channels.fc1', f'{pp_prefix}.mlp_channels.fc1'),
            (f'{th_prefix}.mlp_channels.fc2', f'{pp_prefix}.mlp_channels.fc2'),
        ]
        mapping.extend(layer_mapping)

    return mapping



def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'**SET** {th_name} {th_shape} **TO** {pd_name} {pd_shape}')
        if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
            value = th_params[th_name].data.numpy()
        else:
            value = th_params[th_name].numpy()

        if len(value.shape) == 2 and transpose:
            value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)

    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in torch_model.named_parameters():
        th_params[name] = param

    for name, param in paddle_model.named_buffers():
        pd_params[name] = param
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

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model

    



def main():

    paddle.set_device('cpu')
    paddle_model = build_res_mlp(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    torch_model = timm.create_model('resmlp_big_24_224', pretrained=True)
    torch_model = torch_model.to(device)
    torch_model.eval()
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, 224, 224).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torch_model(x_torch)
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0:100])
    print(out_paddle[0:100])
    assert np.allclose(out_torch, out_paddle, atol = 1e-2)
    
    # save weights for paddle model
    model_path = os.path.join('./resmlp_big_24_224.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)
    print('all done')


if __name__ == "__main__":
    main()
