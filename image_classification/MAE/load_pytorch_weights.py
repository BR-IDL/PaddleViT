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
import timm
from mae_pytorch import models_mae, models_vit
from transformer import build_mae_pretrain as build_model
#from transformer import build_transformer as build_model
from config import *
import random

seed = 0
torch.manual_seed(seed)
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

#model_name = 'mae_vit_huge_patch14'
#config = get_config(f'./configs/vit_huge_patch14_224_pretrain.yaml')

#model_name = 'mae_vit_large_patch16'
#config = get_config(f'./configs/vit_large_patch16_224_pretrain.yaml')

model_name = 'mae_vit_base_patch16'
config = get_config(f'./configs/vit_base_patch16_224_pretrain.yaml')

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
        ('cls_token', f'cls_token'),
        ('mask_token', f'mask_token'),
        ('pos_embed', f'encoder_position_embedding'),
        ('patch_embed.proj', f'patch_embedding.patch_embedding'),
        ('norm', 'encoder.norm'),
        ('decoder_embed', f'linear_projection'),
        ('decoder_pos_embed', f'decoder_position_embedding'),
        ('decoder_norm', f'decoder.norm'),
        ('decoder_pred', f'decoder_pred'),

    ]

    if 'large' in model_name:
        num_enc_layers = 24
        num_dec_layers = 8
    elif 'base' in model_name:
        num_enc_layers = 12
        num_dec_layers = 8
    elif 'huge' in model_name:
        num_enc_layers = 32
        num_dec_layers = 8
    else:
        raise ValueError('now only support large and base model conversion')

    for idx in range(num_enc_layers):
        pp_prefix = f'encoder.layers.{idx}'
        th_prefix = f'blocks.{idx}'
        layer_mapping = [
            (f'{th_prefix}.norm1', f'{pp_prefix}.attn_norm'),
            (f'{th_prefix}.norm2', f'{pp_prefix}.mlp_norm'),
            (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'), 
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'), 
            (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
            (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.out'),
        ]
        mapping.extend(layer_mapping)

    for idx in range(num_dec_layers):
        pp_prefix = f'decoder.layers.{idx}'
        th_prefix = f'decoder_blocks.{idx}'
        layer_mapping = [
            (f'{th_prefix}.norm1', f'{pp_prefix}.attn_norm'),
            (f'{th_prefix}.norm2', f'{pp_prefix}.mlp_norm'),
            (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'), 
            (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'), 
            (f'{th_prefix}.attn.qkv', f'{pp_prefix}.attn.qkv'),
            (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.out'),
        ]
        mapping.extend(layer_mapping)

    #head_mapping = [
    #    #('head', 'classifier')
    #]
    #mapping.extend(head_mapping)

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
    paddle_model = build_model(config)
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    #torch_model = models_vit.__dict__[model_name](global_pool=True)
    torch_model = models_mae.__dict__[model_name](norm_pix_loss=True)
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)
    state_dict = torch.load('./mae_pretrain_vit_base.pth', map_location='cpu')['model']
    print('===========================')
    for key in state_dict:
        print(key)
    print('===========================')
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model = torch_model.to(device)
    torch_model.eval()

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, 224, 224).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    #out_torch = torch_model(x_torch)[1]
    #out_paddle = paddle_model(x_paddle)[1]
    out_torch = torch_model.forward_encoder(x_torch, 0.0)[0]
    out_paddle = paddle_model.forward_encoder(x_paddle, 0.0)[0]

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:100])
    print('========================================================')
    print(out_paddle[0, 0:100])
    assert np.allclose(out_torch, out_paddle, atol = 1e-5)
    
    ## save weights for paddle model
    #model_path = os.path.join(f'./{model_name}.pdparams')
    #paddle.save(paddle_model.state_dict(), model_path)
    #print('all done')


if __name__ == "__main__":
    main()
