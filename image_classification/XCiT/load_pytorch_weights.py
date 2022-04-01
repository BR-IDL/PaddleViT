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
from xcit_torch.xcit import *
from xcit import build_xcit as build_model
from config import get_config


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
    mapping = [
        ('cls_token', 'cls_token'),
        ('pos_embeder.token_projection', 'pos_embeder.token_projection'),
        ('patch_embed.proj.0.0', 'patch_embed.proj.0.0'), # bn
        ('patch_embed.proj.0.1', 'patch_embed.proj.0.1'), # bn
        ('patch_embed.proj.2.0', 'patch_embed.proj.2.0'),
        ('patch_embed.proj.2.1', 'patch_embed.proj.2.1'),
        ('patch_embed.proj.4.0', 'patch_embed.proj.4.0'),
        ('patch_embed.proj.4.1', 'patch_embed.proj.4.1'), # bn
        ('patch_embed.proj.6.0', 'patch_embed.proj.6.0'),
        ('patch_embed.proj.6.1', 'patch_embed.proj.6.1'), # bn
    ]

    for stage_idx in range(config.MODEL.DEPTH):
        pp_prefix = f'blocks.{stage_idx}'
        th_prefix = f'blocks.{stage_idx}'

        layer_mapping = [
            (f'{th_prefix}.gamma1', f'{pp_prefix}.gamma1'),
            (f'{th_prefix}.gamma2', f'{pp_prefix}.gamma2'),
            (f'{th_prefix}.gamma3', f'{pp_prefix}.gamma3'),

            (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
            (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
            (f'{th_prefix}.norm3', f'{pp_prefix}.norm3'),

            (f'{th_prefix}.attn.temperature', f'{pp_prefix}.attn.temperature'),
            (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
            (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),

            (f'{th_prefix}.local_mp.conv1', f'{pp_prefix}.local_mp.conv1'),
            (f'{th_prefix}.local_mp.conv2', f'{pp_prefix}.local_mp.conv2'),
            (f'{th_prefix}.local_mp.bn', f'{pp_prefix}.local_mp.bn'),

            (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
        ]
        mapping.extend(layer_mapping)

    for i in range(2):
        layer_mapping = [
            (f'cls_attn_blocks.{i}.gamma1', f'cls_attn_blocks.{i}.gamma1'),
            (f'cls_attn_blocks.{i}.gamma2', f'cls_attn_blocks.{i}.gamma2'),
            (f'cls_attn_blocks.{i}.norm1', f'cls_attn_blocks.{i}.norm1'),
            (f'cls_attn_blocks.{i}.norm2', f'cls_attn_blocks.{i}.norm2'),
            (f'cls_attn_blocks.{i}.attn.qkv', f'cls_attn_blocks.{i}.attn.qkv'),
            (f'cls_attn_blocks.{i}.attn.proj', f'cls_attn_blocks.{i}.attn.proj'),
            (f'cls_attn_blocks.{i}.mlp.fc1', f'cls_attn_blocks.{i}.mlp.fc1'),
            (f'cls_attn_blocks.{i}.mlp.fc2', f'cls_attn_blocks.{i}.mlp.fc2'),
        ]
        mapping.extend(layer_mapping)

    head_mapping = [
        ('norm', 'norm'),
        ('head', 'head'),
    ]
    mapping.extend(head_mapping)

    return mapping



def convert(torch_model, paddle_model, model_name, config):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'**SET** {th_name} {th_shape} **TO** {pd_name} {pd_shape}')
        if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
            value = th_params[th_name].cpu().data.numpy()
        else:
            value = th_params[th_name].cpu().numpy()

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
            missing = True
            if key.endswith('.weight'):
                if key[:-7] in th_keys:
                    missing = False
            if key.endswith('.bias'):
                if key[:-5] in th_keys:
                    missing = False
            if key.endswith('.running_mean'):
                if key[:-13] in th_keys:
                    missing = False
            if key.endswith('.running_var'):
                if key[:-12] in th_keys:
                    missing = False
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
            if key.endswith('._mean'):
                if key[:-6] in pd_keys:
                    missing = False
            if key.endswith('._variance'):
                if key[:-10] in pd_keys:
                    missing = False
        if missing:
            missing_keys_pd.append(key)


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
            if f'{th_name}.running_mean' in th_params and f'{pd_name}._mean' in pd_params:
                th_name_w = f'{th_name}.running_mean'
                pd_name_w = f'{pd_name}._mean'
                _set_value(th_name_w, pd_name_w)
            if f'{th_name}.running_var' in th_params and f'{pd_name}._variance' in pd_params:
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():
    paddle.set_device('gpu')

    #m_sz = ['nano']
    m_sz = ['nano', 'tiny', 'small', 'medium','large']
    p_sz = ['p8', 'p16']
    sz = [12, 24]
    img_szs = [224, 384]
    dists = [True, False]

    model_name_list = []

    for m in m_sz:
        for p in p_sz:
            for s in sz:
                for img_sz in img_szs:
                  if m == 'nano' and s == 24:
                      continue
                  if m == 'medium' and s == 12:
                      continue
                  if m == 'large' and s == 12:
                      continue
                  name = f'xcit_{m}_{s}_{p}_{img_sz}'
                  model_name_list.append(name)
    print(model_name_list)

    for model_name in model_name_list:
        for dist in dists:
            print(f'============= NOW: {model_name} DIST: {dist} =============')

            if dist:
                if os.path.isfile(os.path.join(f'./{model_name}_dist.pdparams')):
                    print('pdparams exists, skip')
                    continue
            else:
                if os.path.isfile(os.path.join(f'./{model_name}.pdparams')):
                    print('pdparams exists, skip')
                    continue


            sz = int(model_name[-3:])
            if sz == 384 and not dist:
                continue
            config = get_config(f'./configs/{model_name}.yaml')

            #config.defrost()
            #config.TRAIN.DISTILLATION_TYPE = 'none' if not dist else 'hard'
            paddle_model = build_model(config)
    
            paddle_model.eval()
            print_model_named_params(paddle_model)
            print_model_named_buffers(paddle_model)
    
            print('+++++++++++++++++++++++++++++++++++')
            device = torch.device('cuda:1')

            #torch_model = timm.create_model(model_name, pretrained=True)
            torch_model = eval(f'{model_name[:-4]}(pretrained=False)')
            if dist:
                state_dict = torch.load(f'/workspace/pth_models_0330/xcit_pth/{model_name}_dist.pth')
            else:
                state_dict = torch.load(f'/workspace/pth_models_0330/xcit_pth/{model_name}.pth')
            #print(state_dict.keys())
            #print(state_dict['model_ema'].keys())
            #torch_model.load_state_dict(state_dict['model_ema'], strict=True)
            torch_model.load_state_dict(state_dict['model'], strict=True)
            torch_model = torch_model.to(device)
            torch_model.eval()
            print_model_named_params(torch_model)
            print_model_named_buffers(torch_model)
    
            # convert weights
            paddle_model = convert(torch_model, paddle_model, model_name, config)
    
            # check correctness
            x = np.random.randn(1, 3, sz, sz).astype('float32')
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
            if dist:
                model_path = os.path.join(f'./{model_name}_distill.pdparams')
            else:
                model_path = os.path.join(f'./{model_name}.pdparams')

            paddle.save(paddle_model.state_dict(), model_path)
            print(f'{model_name} done')
    print('all done')


if __name__ == "__main__":
    main()
