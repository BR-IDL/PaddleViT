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
from halonet import build_halonet
import os
from config import *
import json

# model_name = 'halonet_50ts_256'
model_name = 'halonet_26t_256'
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
        ('head.fc','classifier.1'),
        ('stem.conv1.conv', 'stem.conv1.conv'),
        ('stem.conv1.bn', 'stem.conv1.bn'),
        ('stem.conv2.conv', 'stem.conv2.conv'),
        ('stem.conv2.bn', 'stem.conv2.bn'),
        ('stem.conv3.conv', 'stem.conv3.conv'),
        ('stem.conv3.bn', 'stem.conv3.bn'),
    ]

    # torch 'layers' to  paddle 'stages'
    # depths = config.MODEL.TRANS.STAGE_DEPTHS
    depths = [3,4,6,3]
    num_stages = len(depths)
    for stage_idx in range(num_stages):
        th_s_prefix = f'stages.{stage_idx}'
        pp_s_prefix = f'stage{stage_idx+1}.blocks'
        for block_idx in range(depths[stage_idx]):
            th_b_prefix = f'{th_s_prefix}.{block_idx}'
            pp_b_prefix = f'{pp_s_prefix}.{block_idx}'
            layer_mapping = [
                (f'{th_b_prefix}.conv1_1x1.conv', f'{pp_b_prefix}.conv1_1x1.conv'),
                (f'{th_b_prefix}.conv1_1x1.bn', f'{pp_b_prefix}.conv1_1x1.bn'),
                (f'{th_b_prefix}.conv2_kxk.conv', f'{pp_b_prefix}.conv2_kxk.conv'),
                (f'{th_b_prefix}.conv2_kxk.bn', f'{pp_b_prefix}.conv2_kxk.bn'),
                (f'{th_b_prefix}.conv3_1x1.conv', f'{pp_b_prefix}.conv3_1x1.conv'),
                (f'{th_b_prefix}.conv3_1x1.bn', f'{pp_b_prefix}.conv3_1x1.bn'),
                (f'{th_b_prefix}.shortcut.conv', f'{pp_b_prefix}.creat_shortcut.conv'),
                (f'{th_b_prefix}.shortcut.bn', f'{pp_b_prefix}.creat_shortcut.bn'),
                (f'{th_b_prefix}.self_attn.q', f'{pp_b_prefix}.self_attn.q'),
                (f'{th_b_prefix}.self_attn.kv', f'{pp_b_prefix}.self_attn.kv'),
                (f'{th_b_prefix}.self_attn.pos_embed.height_rel', f'{pp_b_prefix}.self_attn.pos_embed.rel_height'),
                (f'{th_b_prefix}.self_attn.pos_embed.width_rel', f'{pp_b_prefix}.self_attn.pos_embed.rel_width'),
                (f'{th_b_prefix}.post_attn', f'{pp_b_prefix}.post_attn.bn'),
            ]
            mapping.extend(layer_mapping)

    return mapping



def convert(torch_model, paddle_model):
    new_pd_params = []
    def _set_value(th_name, pd_name, no_transpose=False):
        if (pd_name == 'classifier.1.weight'):
            th_shape = th_params[th_name].shape
            pd_shape = tuple(pd_params[pd_name].shape)
        if(th_name in th_params.keys() and pd_name in pd_params.keys()):
            th_shape = th_params[th_name].shape
            pd_shape = tuple(pd_params[pd_name].shape)  # paddle shape default type is list
            # assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
            print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
            value = th_params[th_name].data.numpy()
            if len(value.shape) == 2:
                if not no_transpose:
                    value = value.transpose((1, 0))
            pd_params[pd_name].set_value(value)
            new_pd_params.append(pd_name)
        else:
            print('%s not in th_params'%(th_name))
            print('%s not in pd_params'%(pd_name))

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
        if th_name in th_params.keys():  # nn.Parameters
            if th_name.endswith('height_rel'):
                _set_value(th_name, pd_name, no_transpose=True)
            elif th_name.endswith('width_rel'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
                _set_value(th_name, pd_name)
        else:  # weight & bias
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_mean' in th_params.keys():
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model,new_pd_params


def main():
    paddle.set_device('cpu')
    paddle_model = build_halonet(config)
    paddle_model.eval()

    print(paddle_model)
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    # torch_model = timm.create_model('halonet50ts',pretrained=True)
    torch_model = timm.create_model('halonet26t', pretrained=True)
    torch_model = torch_model.to(device)
    torch_model.eval()

    print(torch_model)
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)
    print('+++++++++++++++++++++++++++++++++++')

    # convert weights
    paddle_model,new_pd_params_list = convert(torch_model, paddle_model)

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
    print(np.sum(out_torch),np.sum(out_paddle))

    assert np.allclose(out_torch, out_paddle, atol = 1e-5)
    
    # save weights for paddle model
    model_path = os.path.join(f'./{model_name}.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)
    print('all done')


if __name__ == "__main__":
    main()
