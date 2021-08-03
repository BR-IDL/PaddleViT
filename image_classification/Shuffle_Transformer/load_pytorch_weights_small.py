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
from shuffle_transformer import *
from shuffle_pth.shuffle_transformer_torch import ShuffleTransformer as ShuffleTransformerTorch
from config import *

config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg', type=str, default='./configs/shuffle_vit_small_patch4_window7_224.yaml')
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-image_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-eval', action="store_true")
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
args = parser.parse_args()

config = get_config()
config = update_config(config, args)
print(config)


def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)


def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)


def torch_to_paddle_mapping():
    # (torch_param_name, paddle_param_name)
    mapping = [
        ('to_token.conv1.0', 'patch_embedding.conv1.0'), # conv
        ('to_token.conv1.1', 'patch_embedding.conv1.1'), # bn
        ('to_token.conv2.0', 'patch_embedding.conv2.0'), # conv
        ('to_token.conv2.1', 'patch_embedding.conv2.1'), # bn
        ('to_token.conv3', 'patch_embedding.conv3'), # conv
    ]

    for stage_idx, num_layers in enumerate(config.MODEL.TRANS.DEPTHS):
        for idx in range(num_layers):
            th_layer_idx_0 = idx // 2
            th_layer_idx_1 = idx % 2
            th_prefix = f'stage{stage_idx+1}.layers.{th_layer_idx_0}.{th_layer_idx_1}'
            pp_prefix = f'stages.{stage_idx}.layers.{idx}'
            layer_mapping = [
                (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'), #bn
                (f'{th_prefix}.attn.relative_position_bias_table', f'{pp_prefix}.attn.relative_position_bias_table'), # no transpose
                (f'{th_prefix}.attn.relative_position_index', f'{pp_prefix}.attn.relative_position_index'), # no transpose
                (f'{th_prefix}.attn.to_qkv', f'{pp_prefix}.attn.qkv'),
                (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
                (f'{th_prefix}.local', f'{pp_prefix}.local'),
                (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'), #bn
                (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'), 
                (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'), 
                (f'{th_prefix}.norm3', f'{pp_prefix}.norm3'), #bn
            ]
            mapping.extend(layer_mapping)

            if stage_idx > 0:
                layer_mapping = [
                    (f'stage{stage_idx+1}.patch_partition.norm', f'stages.{stage_idx}.patch_partition.norm'), #bn
                    (f'stage{stage_idx+1}.patch_partition.reduction', f'stages.{stage_idx}.patch_partition.reduction'),
                ]
                mapping.extend(layer_mapping)

    head_mapping = [
        ('head', 'head'),
    ]
    mapping.extend(head_mapping)

    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
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
    for name, param in paddle_model.named_buffers():
        pd_params[name] = param

    for name, param in torch_model.named_parameters():
        th_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()
    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            if 'relative_position' in th_name:
                _set_value(th_name, pd_name, transpose=False)
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

            if f'{th_name}.running_mean' in th_params.keys():
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():

    paddle.set_device('cpu')
    paddle_model = build_shuffle_transformer(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print('--------------')
    print_model_named_buffers(paddle_model)
    print('----------------------------------')

    device = torch.device('cpu')
    torch_model = ShuffleTransformerTorch(layers=[2, 2, 18, 2],
                                          num_heads=[3, 6, 12, 24],
                                          qkv_bias=True,
                                          embed_dim=96,
                                          )
    model_state_dict = torch.load('./shuffle_pth/shuffle_vit_small_patch4_window7_224_ep292.pth', map_location='cpu') 
    torch_model.load_state_dict(model_state_dict['model'])
    torch_model = torch_model.to(device)
    torch_model.eval()

    print_model_named_params(torch_model)
    print('--------------')
    print_model_named_buffers(torch_model)
    print('----------------------------------')


    #return

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
    print(out_torch[0, 0:100])
    print(out_paddle[0, 0:100])
    assert np.allclose(out_torch, out_paddle, atol = 1e-5)
    
    # save weights for paddle model
    model_path = os.path.join('./shuffle_vit_small_patch4_window7_224.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
