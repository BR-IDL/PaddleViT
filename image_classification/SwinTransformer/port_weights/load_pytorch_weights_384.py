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
from swin_transformer import *
from config import *


config = get_config('./configs/swin_base_patch4_window12_384.yaml')
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
        ('patch_embed.proj', 'patch_embedding.patch_embed'),
        ('patch_embed.norm', 'patch_embedding.norm'),
    ]
    
    # torch 'layers' to  paddle 'stages'
    depths = config.MODEL.TRANS.STAGE_DEPTHS
    num_stages = len(depths)
    for stage_idx in range(num_stages):
        pp_s_prefix = f'stages.{stage_idx}.blocks'
        th_s_prefix = f'layers.{stage_idx}.blocks'
        for block_idx in range(depths[stage_idx]):
            th_b_prefix = f'{th_s_prefix}.{block_idx}'
            pp_b_prefix = f'{pp_s_prefix}.{block_idx}'
            layer_mapping = [
                (f'{th_b_prefix}.norm1', f'{pp_b_prefix}.norm1'),
                (f'{th_b_prefix}.attn.relative_position_bias_table', f'{pp_b_prefix}.attn.relative_position_bias_table'),
                (f'{th_b_prefix}.attn.qkv', f'{pp_b_prefix}.attn.qkv'),
                (f'{th_b_prefix}.attn.proj', f'{pp_b_prefix}.attn.proj'),
                (f'{th_b_prefix}.norm2', f'{pp_b_prefix}.norm2'),
                (f'{th_b_prefix}.mlp.fc1', f'{pp_b_prefix}.mlp.fc1'),
                (f'{th_b_prefix}.mlp.fc2', f'{pp_b_prefix}.mlp.fc2'),
            ]
            mapping.extend(layer_mapping)
        # stage downsample: last stage does not have downsample ops
        if stage_idx < num_stages - 1:
            mapping.extend([
                (f'layers.{stage_idx}.downsample.reduction.weight', f'stages.{stage_idx}.downsample.reduction.weight'),
                (f'layers.{stage_idx}.downsample.norm', f'stages.{stage_idx}.downsample.norm')])

    mapping.extend([
        ('norm', 'norm'),
        ('head', 'fc')])
    return mapping



def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, no_transpose=False):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
        if len(value.shape) == 2:
            if not no_transpose:
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
            if th_name.endswith('relative_position_bias_table'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
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
    paddle_model = build_swin(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    torch_model = timm.create_model('swin_base_patch4_window12_384', pretrained=True)
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
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:20])
    print(out_paddle[0, 0:20])
    assert np.allclose(out_torch, out_paddle, atol = 1e-4)
    
    # save weights for paddle model
    model_path = os.path.join('./swin_base_patch4_window12_384.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)



    #tmp = np.random.randn(1, 56, 128, 128).astype('float32')
    #xp = paddle.to_tensor(tmp)
    #xt = torch.Tensor(tmp).to(device)
    #xps = paddle.roll(xp, shifts=(-3, -3), axis=(1,2))
    #xts = torch.roll(xt,shifts=(-3, -3), dims=(1,2))
    #xps = xps.cpu().numpy()
    #xts = xts.data.cpu().numpy()
    #assert np.allclose(xps, xts, atol=1e-4)

if __name__ == "__main__":
    main()
