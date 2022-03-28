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
"""convert pytorch model weights to paddle pdparams"""
import os
import numpy as np
import paddle
import torch
import timm
from crossvit import build_crossvit as build_model
from config import get_config

from CrossViT_torch.models.crossvit import *


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


def torch_to_paddle_mapping(model_name, config):
    if 'dagger' in model_name:
        mapping = [
            ('patch_embed.0.proj.0', 'patch_embed.0.proj.0'),
            ('patch_embed.1.proj.0', 'patch_embed.1.proj.0'),
            ('patch_embed.0.proj.2', 'patch_embed.0.proj.2'),
            ('patch_embed.1.proj.2', 'patch_embed.1.proj.2'),
            ('patch_embed.0.proj.4', 'patch_embed.0.proj.4'),
            ('patch_embed.1.proj.4', 'patch_embed.1.proj.4'),
            ('pos_embed.0', 'pos_embed.0'),    
            ('cls_token.0', 'cls_token.0'),    
            ('pos_embed.1', 'pos_embed.1'),    
            ('cls_token.1', 'cls_token.1'),    
        ] 
    else:
        mapping = [
            ('patch_embed.0.proj', 'patch_embed.0.proj'),
            ('patch_embed.1.proj', 'patch_embed.1.proj'),
            ('pos_embed.0', 'pos_embed.0'),    
            ('cls_token.0', 'cls_token.0'),    
            ('pos_embed.1', 'pos_embed.1'),    
            ('cls_token.1', 'cls_token.1'),    
        ] 

    # torch 'layers' to  paddle 'stages'
    depths = config.MODEL.DEPTH
    num_stages = len(depths)
    for stage_idx in range(num_stages):
        pp_s_prefix = f'blocks.{stage_idx}'
        th_s_prefix = f'blocks.{stage_idx}'
        for block_idx, depth in enumerate(depths[stage_idx]):
            for d in range(depth):
                th_b_prefix = f'{th_s_prefix}.blocks.{block_idx}.{d}'
                pp_b_prefix = f'{pp_s_prefix}.blocks.{block_idx}.{d}'
                layer_mapping = [
                    (f'{th_b_prefix}.norm1', f'{pp_b_prefix}.norm1'),
                    (f'{th_b_prefix}.attn.qkv', f'{pp_b_prefix}.attn.qkv'),
                    (f'{th_b_prefix}.attn.proj', f'{pp_b_prefix}.attn.out'),
                    (f'{th_b_prefix}.norm2', f'{pp_b_prefix}.norm2'),
                    (f'{th_b_prefix}.mlp.fc1', f'{pp_b_prefix}.mlp.fc1'),
                    (f'{th_b_prefix}.mlp.fc2', f'{pp_b_prefix}.mlp.fc2'),
                ]
                mapping.extend(layer_mapping)   

        layer_mapping_2 = [
            (f'{th_s_prefix}.projs.0.0',          f'{pp_s_prefix}.projs.0.0'),
            (f'{th_s_prefix}.projs.0.2',          f'{pp_s_prefix}.projs.0.2'),
            (f'{th_s_prefix}.projs.1.0',          f'{pp_s_prefix}.projs.1.0'),
            (f'{th_s_prefix}.projs.1.2',          f'{pp_s_prefix}.projs.1.2'),
            (f'{th_s_prefix}.fusion.0.norm1',     f'{pp_s_prefix}.fusion.0.norm1'),
            (f'{th_s_prefix}.fusion.0.attn.wq',   f'{pp_s_prefix}.fusion.0.attn.wq'),
            (f'{th_s_prefix}.fusion.0.attn.wk',   f'{pp_s_prefix}.fusion.0.attn.wk'),
            (f'{th_s_prefix}.fusion.0.attn.wv',   f'{pp_s_prefix}.fusion.0.attn.wv'),
            (f'{th_s_prefix}.fusion.0.attn.proj', f'{pp_s_prefix}.fusion.0.attn.proj'),
            (f'{th_s_prefix}.fusion.1.norm1',     f'{pp_s_prefix}.fusion.1.norm1'),
            (f'{th_s_prefix}.fusion.1.attn.wq',   f'{pp_s_prefix}.fusion.1.attn.wq'),
            (f'{th_s_prefix}.fusion.1.attn.wk',   f'{pp_s_prefix}.fusion.1.attn.wk'),
            (f'{th_s_prefix}.fusion.1.attn.wv',   f'{pp_s_prefix}.fusion.1.attn.wv'),
            (f'{th_s_prefix}.fusion.1.attn.proj', f'{pp_s_prefix}.fusion.1.attn.proj'),
            (f'{th_s_prefix}.revert_projs.0.0',   f'{pp_s_prefix}.revert_projs.0.0'),
            (f'{th_s_prefix}.revert_projs.0.2',   f'{pp_s_prefix}.revert_projs.0.2'),
            (f'{th_s_prefix}.revert_projs.1.0',   f'{pp_s_prefix}.revert_projs.1.0'),
            (f'{th_s_prefix}.revert_projs.1.2',   f'{pp_s_prefix}.revert_projs.1.2'),
        ]
        mapping.extend(layer_mapping_2)   

    mapping.extend([
        ('head.0', 'head.0'),
        ('head.1', 'head.1'),
        ('norm.0', 'norm.0'),
        ('norm.1', 'norm.1'),
    ])
    return mapping


def convert(torch_model, paddle_model, model_name, config):
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
    mapping = torch_to_paddle_mapping(model_name, config)


    missing_keys_th = []
    missing_keys_pd = []
    zip_map = list(zip(*mapping))
    th_keys = list(zip_map[0])
    pd_keys = list(zip_map[1])

    for key in th_params:
        missing = False
        if key not in th_keys:
            if key.endswith('.weight'):
                if key[:-7] not in th_keys:
                    missing = True
            if key.endswith('.bias'):
                if key[:-5] not in th_keys:
                    missing = True
        if missing:
            missing_keys_th.append(key)

    for key in pd_params:
        missing = False
        if key not in pd_keys:
            missing = True
            if key.endswith('.weight'):
                if key[:-7] in pd_keys:
                    missing = False
            if key.endswith('.bias'):
                if key[:-5] in pd_keys:
                    missing = False
        if missing:
            missing_keys_th.append(key)


    print('====================================')
    print('missing_keys_pytorch:')
    print(missing_keys_th)
    print('missing_keys_paddle:')
    print(missing_keys_pd)
    print('====================================')

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params and pd_name in pd_params: # nn.Parameters
            _set_value(th_name, pd_name)
        else:
            if f'{th_name}.weight' in th_params and f'{pd_name}.weight' in pd_params:
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)
            if f'{th_name}.bias' in th_params and f'{pd_name}.bias' in pd_params:
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():
    paddle.set_device('cpu')
    model_name_list = [
        "crossvit_15_224",
        "crossvit_15_dagger_224",
        "crossvit_15_dagger_384",
        "crossvit_18_224",
        "crossvit_18_dagger_224",
        "crossvit_18_dagger_384",
        "crossvit_9_224",
        "crossvit_9_dagger_224",
        "crossvit_base_224",
        "crossvit_small_224",
        "crossvit_tiny_224",
    ]

    for model_name in model_name_list:
        print(f'============= NOW: {model_name} =============')
        if '224' in model_name or '384' in model_name:
            sz = int(model_name[-3::])
        else:
            sz = 224
        config = get_config(f'./configs/{model_name}.yaml')
        paddle_model = build_model(config)

        paddle_model.eval()
        print_model_named_params(paddle_model)
        print_model_named_buffers(paddle_model)

        print('+++++++++++++++++++++++++++++++++++')
        device = torch.device('cpu')
        #torch_model = timm.create_model(model_name, pretrained=True)

        torch_model = eval(f"{model_name}(pretrained=True)") 
        torch_model = torch_model.to(device)
        torch_model.eval()
        print_model_named_params(torch_model)
        print_model_named_buffers(torch_model)

        # convert weights
        paddle_model = convert(torch_model, paddle_model, model_name, config)

        # check correctness
        x = np.random.randn(2, 3, sz, sz).astype('float32')
        x_paddle = paddle.to_tensor(x)
        x_torch = torch.Tensor(x).to(device)

        out_torch = torch_model(x_torch)
        out_paddle = paddle_model(x_paddle)

        out_torch = out_torch.data.cpu().numpy()
        out_paddle = out_paddle.cpu().numpy()

        print(out_torch.shape, out_paddle.shape)
        print(out_torch[0, 0:100])
        print('========================================================')
        print(out_paddle[0, 0:100])
        assert np.allclose(out_torch, out_paddle, atol = 1e-4)

        # save weights for paddle model
        model_path = os.path.join(f'./{model_name}.pdparams')
        paddle.save(paddle_model.state_dict(), model_path)
        print(f'{model_name} done')
    print('all done')


if __name__ == "__main__":
    main()
