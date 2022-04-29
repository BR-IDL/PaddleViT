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
from levit import build_levit as build_model
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
        ('patch_embed.0.c', 'patch_embed.0.conv'),
        ('patch_embed.0.bn', 'patch_embed.0.norm'),
        ('patch_embed.2.c', 'patch_embed.2.conv'),
        ('patch_embed.2.bn', 'patch_embed.2.norm'),
        ('patch_embed.4.c', 'patch_embed.4.conv'),
        ('patch_embed.4.bn', 'patch_embed.4.norm'),
        ('patch_embed.6.c', 'patch_embed.6.conv'),
        ('patch_embed.6.bn', 'patch_embed.6.norm'),
    ]

    th_idx = 0
    pp_idx = 0
    for stage_idx, stage_depth in enumerate(config.MODEL.DEPTH):
        for block_idx in range(stage_depth):
            th_prefix = f'blocks.{th_idx}'
            pp_prefix = f'blocks.{pp_idx}'
            layer_mapping = [
                (f'{th_prefix}.m.attention_biases', f'{pp_prefix}.attn.attn_biases'),
                (f'{th_prefix}.m.attention_bias_idxs', f'{pp_prefix}.attn.attn_bias_idxs'),
                (f'{th_prefix}.m.qkv.c', f'{pp_prefix}.attn.qkv.linear'),
                (f'{th_prefix}.m.qkv.bn', f'{pp_prefix}.attn.qkv.norm'),
                (f'{th_prefix}.m.proj.1.c', f'{pp_prefix}.attn.proj.1.linear'),
                (f'{th_prefix}.m.proj.1.bn', f'{pp_prefix}.attn.proj.1.norm'),
            ]
            mapping.extend(layer_mapping)

            th_idx += 1 
            th_prefix = f'blocks.{th_idx}'
            layer_mapping = [
                (f'{th_prefix}.m.0.c', f'{pp_prefix}.mlp.ln1.linear'),
                (f'{th_prefix}.m.0.bn', f'{pp_prefix}.mlp.ln1.norm'),
                (f'{th_prefix}.m.2.c', f'{pp_prefix}.mlp.ln2.linear'),
                (f'{th_prefix}.m.2.bn', f'{pp_prefix}.mlp.ln2.norm'),
            ]
            mapping.extend(layer_mapping)
            pp_idx += 1
            th_idx += 1

        if stage_idx < len(config.MODEL.DEPTH) - 1:
            # sub sample block
            th_prefix = f'blocks.{th_idx}'
            pp_prefix = f'blocks.{pp_idx}'
            layer_mapping = [
                (f'{th_prefix}.attention_biases', f'{pp_prefix}.attn.attn_biases'),
                (f'{th_prefix}.attention_bias_idxs', f'{pp_prefix}.attn.attn_bias_idxs'),
                (f'{th_prefix}.kv.c', f'{pp_prefix}.attn.kv.linear'),
                (f'{th_prefix}.kv.bn', f'{pp_prefix}.attn.kv.norm'),
                (f'{th_prefix}.q.1.c', f'{pp_prefix}.attn.q.1.linear'),
                (f'{th_prefix}.q.1.bn', f'{pp_prefix}.attn.q.1.norm'),
                (f'{th_prefix}.proj.1.c', f'{pp_prefix}.attn.proj.1.linear'),
                (f'{th_prefix}.proj.1.bn', f'{pp_prefix}.attn.proj.1.norm'),
            ]
            mapping.extend(layer_mapping)

            th_idx += 1 
            th_prefix = f'blocks.{th_idx}'
            layer_mapping = [
                (f'{th_prefix}.m.0.c', f'{pp_prefix}.mlp.ln1.linear'),
                (f'{th_prefix}.m.0.bn', f'{pp_prefix}.mlp.ln1.norm'),
                (f'{th_prefix}.m.2.c', f'{pp_prefix}.mlp.ln2.linear'),
                (f'{th_prefix}.m.2.bn', f'{pp_prefix}.mlp.ln2.norm'),
            ]
            mapping.extend(layer_mapping)
            pp_idx += 1
            th_idx += 1

    head_mapping = [
        ('head.bn', 'head.norm'),
        ('head.l', 'head.linear'),
    ]
    mapping.extend(head_mapping)
    
    if config.TRAIN.DISTILLATION_TYPE != 'none':
        head_mapping = [
            ('head_dist.bn', 'head_dist.norm'),
            ('head_dist.l', 'head_dist.linear'),
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
        if key.endswith('num_batches_tracked'):
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
            if 'attention_biases' in th_name or 'attention_bias_idxs' in th_name:
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
                th_name_w = f'{th_name}.running_mean'
                pd_name_w = f'{pd_name}._mean'
                _set_value(th_name_w, pd_name_w)
            if f'{th_name}.running_var' in th_params and f'{pd_name}._variance' in pd_params:
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():
    paddle.set_device('cpu')
    model_name_list = [
        "levit_128s",
        "levit_128",
        "levit_192",
        "levit_256",
        "levit_384",
    ]

    for model_name in model_name_list:
        print(f'============= NOW: {model_name} =============')
        sz = 224
        config = get_config(f'./configs/{model_name}.yaml')

        paddle_model = build_model(config)

        paddle_model.eval()
        print_model_named_params(paddle_model)
        print_model_named_buffers(paddle_model)
        #print(paddle_model)

        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        device = torch.device('cpu')

        torch_model = timm.create_model(model_name, pretrained=True)
        torch_model.eval()
        torch_model = torch_model.to(device)
        print_model_named_params(torch_model)
        print_model_named_buffers(torch_model)

        #print(torch_model)

        # convert weights
        paddle_model = convert(torch_model, paddle_model, model_name, config)

        # check correctness
        x = np.random.randn(2, 3, sz, sz).astype('float32')
        x_paddle = paddle.to_tensor(x)
        x_torch = torch.Tensor(x).to(device)

        out_torch = torch_model(x_torch)
        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        out_paddle = paddle_model(x_paddle)

        out_torch = out_torch.data.cpu().numpy()
        out_paddle = out_paddle.cpu().numpy()

        print(out_torch.shape, out_paddle.shape)
        print(out_torch[0, 0:100])
        print('========================================================')
        print(out_paddle[0, 0:100])
        assert np.allclose(out_torch, out_paddle, atol = 1e-5)

        # save weights for paddle model
        model_path = os.path.join(f'./{model_name}.pdparams')
        paddle.save(paddle_model.state_dict(), model_path)
        print(f'{model_name} done')
    print('all done')


if __name__ == "__main__":
    main()
