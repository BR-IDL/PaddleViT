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
from config import get_config
from t2t_vit import build_t2t_vit as build_model

from T2T_ViT_torch.models.t2t_vit import *
from T2T_ViT_torch.utils import load_for_transfer_learning


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
 	# (torch_param_name, paddle_param_name)
    mapping = [
        ('cls_token', 'cls_token'),
        ('pos_embed', 'pos_embed'),
    ]

    for idx in range(1, 3):
        th_prefix = f'tokens_to_token.attention{idx}'
        pp_prefix = f'patch_embed.attn{idx}'
        if '_t_' in model_name:
        	layer_mapping = [
        	    (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
        	    (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
        	    (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
        	    (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
        	    (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
        	    (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
        	]
        else:
            layer_mapping = [
                (f'{th_prefix}.w', f'{pp_prefix}.w'),
                (f'{th_prefix}.kqv', f'{pp_prefix}.kqv'),
                (f'{th_prefix}.proj', f'{pp_prefix}.proj'),
                (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                (f'{th_prefix}.mlp.0', f'{pp_prefix}.mlp.0'),
                (f'{th_prefix}.mlp.2', f'{pp_prefix}.mlp.2'),
            ]
        mapping.extend(layer_mapping)
    mapping.append(('tokens_to_token.project','patch_embed.proj'))


    num_layers = config.MODEL.DEPTH
    for idx in range(num_layers):
        th_prefix = f'blocks.{idx}'
        pp_prefix = f'blocks.{idx}'
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
            if th_name.endswith('w'):
                _set_value(th_name, pd_name, transpose=False)
            else:
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
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)
            if f'{th_name}.running_var' in th_params and f'{pd_name}._variance' in pd_params:
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():
    paddle.set_device('cpu')
    model_name_list = ['t2t_vit_7',
                       't2t_vit_10',
                       't2t_vit_12',
                       't2t_vit_14',
                       't2t_vit_14_384',
                       't2t_vit_19',
                       't2t_vit_24',
                       't2t_vit_24_token_labeling',
                       't2t_vit_t_14',
                       't2t_vit_t_19',
                       't2t_vit_t_24']
    pth_model_path_list = ['./T2T_ViT_torch/t2t-vit-pth-models/71.7_T2T_ViT_7.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/75.2_T2T_ViT_10.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/76.5_T2T_ViT_12.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/81.5_T2T_ViT_14.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/83.3_T2T_ViT_14.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/81.9_T2T_ViT_19.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/82.3_T2T_ViT_24.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/84.2_T2T_ViT_24.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/81.7_T2T_ViTt_14.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/82.4_T2T_ViTt_19.pth.tar',
                      './T2T_ViT_torch/t2t-vit-pth-models/82.6_T2T_ViTt_24.pth.tar']

    for model_name, pth_model_path in zip(model_name_list, pth_model_path_list):
        print(f'============= NOW: {model_name} =============')
        sz = 384 if '384' in model_name else 224

        if 'token_labeling' in model_name:
            config = get_config(f'./configs/{model_name[:-15]}.yaml')
        else:
            config = get_config(f'./configs/{model_name}.yaml')
        paddle_model = build_model(config)

        paddle_model.eval()
        print_model_named_params(paddle_model)
        print_model_named_buffers(paddle_model)

        print('+++++++++++++++++++++++++++++++++++')
        device = torch.device('cpu')
        if 'token_labeling' in model_name:
            torch_model = eval(f'{model_name[:-15]}(img_size={sz})')
        else:
            if '384' in model_name:
                torch_model = eval(f'{model_name[:-4]}(img_size={sz})')
            else:
                torch_model = eval(f'{model_name}(img_size={sz})')

        load_for_transfer_learning(torch_model,
                                   pth_model_path,
                                   use_ema=True,
                                   strict=False,
                                   num_classes=1000)
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
        assert np.allclose(out_torch, out_paddle, atol = 1e-2)

        # save weights for paddle model
        model_path = os.path.join(f'./{model_name}.pdparams')
        paddle.save(paddle_model.state_dict(), model_path)
        print(f'{model_name} done')
    print('all done')


if __name__ == "__main__":
    main()
