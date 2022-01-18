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

import argparse
import numpy as np
import paddle
import torch
from transformer import *
from config import *


config = get_config('./configs/vit_base_patch16_224.yaml')
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
        ('patch_embed.proj', 'patch_embedding.patch_embedding'),
        ('cls_token', 'patch_embedding.cls_token'),
        ('pos_embed', 'patch_embedding.position_embeddings'),
    ]
    
    # torch 'blocks' to  paddle 'layers'
    depth = config.MODEL.TRANS.DEPTH
    for idx in range(depth):
        th_prefix = f'blocks.{idx}'
        pp_prefix = f'encoder.layers.{idx}'
        layer_mapping = [
            (f'{th_prefix}.norm1', f'{pp_prefix}.attn_norm'),
            (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
            (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
            (f'{th_prefix}.norm2', f'{pp_prefix}.mlp_norm'),
            (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
        ]
        mapping.extend(layer_mapping)

    mapping.extend([('norm', 'encoder.encoder_norm')])
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
    paddle_model = build_vit(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')

    device = torch.device('cpu')
    torch_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
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
    
    ## save weights for paddle model
    #model_path = os.path.join('./vit_base_patch16_224.pdparams')
    #paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
