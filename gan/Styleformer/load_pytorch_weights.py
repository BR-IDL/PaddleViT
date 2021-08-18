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
import os
import numpy as np
import paddle
import torch
from training.networks_Generator import *
import legacy
import dnnlib
from generator import Generator
from config import *


config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg', type=str, default='./configs/styleformer_cifar10.yaml')
parser.add_argument('-dataset', type=str, default="cifar10")
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-image_size', type=int, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-data_path', type=str, default='/dataset/cifar10/')
parser.add_argument('-eval', action="store_true")
parser.add_argument('-pretrained', type=str, default=None)
args = parser.parse_args()

config = get_config()
config = update_config(config, args)
print(config)


def print_model_named_params(model):
    sum=0
    print('----------------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)
        sum=sum+1
    print(sum)
    print('----------------------------------')


def print_model_named_buffers(model):
    sum=0
    print('----------------------------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
        sum=sum+1
    print(sum)
    print('----------------------------------')


def torch_to_paddle_mapping():
    resolution = config.MODEL.GEN.RESOLUTION
    prefix = f'synthesis.b{resolution}_0'
    mapping = [
        (f'{prefix}.const', f'{prefix}.const'),
    ]
    num_layers = config.MODEL.GEN.NUM_LAYERS
    # torch 'layers' to  paddle 'stages'
    num_stages = len(num_layers)
    linformer = config.MODEL.GEN.LINFORMER
    i = 0
    for i in range(num_stages):
        stage_idx = 2**i * resolution
        pp_s_prefix = f'synthesis.b{stage_idx}_'
        th_s_prefix = f'synthesis.b{stage_idx}_'
        mapping.extend([(f'{th_s_prefix}0.pos_embedding', f'{pp_s_prefix}0.pos_embedding')])

        for block_idx in range(num_layers[i]):
            th_b_prefix = f'{th_s_prefix}{block_idx}'
            pp_b_prefix = f'{pp_s_prefix}{block_idx}'
            layer_mapping = [
                (f'{th_b_prefix}.enc.q_weight', f'{pp_b_prefix}.enc.q_weight'),
                (f'{th_b_prefix}.enc.k_weight', f'{pp_b_prefix}.enc.k_weight'),
                (f'{th_b_prefix}.enc.v_weight', f'{pp_b_prefix}.enc.v_weight'),
                (f'{th_b_prefix}.enc.w_weight', f'{pp_b_prefix}.enc.w_weight'),
                (f'{th_b_prefix}.enc.u_weight', f'{pp_b_prefix}.enc.u_weight'),
                (f'{th_b_prefix}.enc.bias', f'{pp_b_prefix}.enc.bias'),
                (f'{th_b_prefix}.enc.affine1.weight', f'{pp_b_prefix}.enc.affine1.weight'),
                (f'{th_b_prefix}.enc.affine1.bias', f'{pp_b_prefix}.enc.affine1.bias'),
                (f'{th_b_prefix}.resample_filter', f'{pp_b_prefix}.resample_filter'),
                (f'{th_b_prefix}.enc.noise_const', f'{pp_b_prefix}.enc.noise_const'),
                (f'{th_b_prefix}.enc.noise_strength', f'{pp_b_prefix}.enc.noise_strength'),
            ]
            if stage_idx>=32 and linformer:
                mapping.extend([(f'{th_s_prefix}0.proj_weight', f'{pp_s_prefix}0.proj_weight')])
            mapping.extend(layer_mapping)

        mapping.extend([
            (f'{th_b_prefix}.torgb.weight', f'{pp_b_prefix}.torgb.weight'),
            (f'{th_b_prefix}.torgb.bias', f'{pp_b_prefix}.torgb.bias'),
            (f'{th_b_prefix}.torgb.affine.weight', f'{pp_b_prefix}.torgb.affine.weight'),
            (f'{th_b_prefix}.torgb.affine.bias', f'{pp_b_prefix}.torgb.affine.bias'),
        ])
        i = i + 1
    mapping.extend([('mapping.fc0', 'mapping.fc0'),
                    ('mapping.fc1', 'mapping.fc1'),
                    ('mapping.w_avg', 'mapping.w_avg')])
    return mapping


def convert(torch_model, paddle_model):

    def _set_value(th_name, pd_name, no_transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
            value = th_params[th_name].data.numpy()
        else:
            value = th_params[th_name].numpy()
        if value.shape == ():
            value = value.reshape(1)
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

    for name, buff in paddle_model.named_buffers():
        pd_params[name] = buff
    for name, buff in torch_model.named_buffers():
        th_params[name] = buff

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()
    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            if th_name.endswith('relative_position_bias_table'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
                _set_value(th_name, pd_name, no_transpose=True)
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
    paddle_model = Generator(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    device = torch.device('cpu')
    # load weights from local
    torch_model = Generator_torch(z_dim=512,c_dim=0,w_dim=512,img_resolution=32,img_channels=3)
    with dnnlib.util.open_url(r'./Pretrained_CIFAR10.pkl') as f:
        torch_model = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    torch_model.eval()

    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)
    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(32, 512).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torch_model(x_torch, c=torch.zeros(1))
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    out_paddle = paddle_model(x_paddle, c=paddle.zeros([1]))

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:20])
    print(out_paddle[0, 0:20])
    assert np.allclose(out_torch, out_paddle, atol = 1e-2)

    # save weights for paddle model
    model_path = os.path.join('./cifar10.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
