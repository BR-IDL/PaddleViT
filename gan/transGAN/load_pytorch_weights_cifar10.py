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

"""load weight """

import torch
import torch.nn as nn
import paddle
import argparse

import models_search
from models.ViT_custom import Generator
from models.ViT_custom_scale2 import Discriminator

from config import get_config, update_config

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
    py_prefix = 'module'
    mapping = [
        (f'{py_prefix}.pos_embed_1', 'pos_embed_1'),
        (f'{py_prefix}.pos_embed_2', 'pos_embed_2'),
        (f'{py_prefix}.pos_embed_3', 'pos_embed_3'),
        (f'{py_prefix}.l1.weight', 'l1.weight'),
        (f'{py_prefix}.l1.bias', 'l1.bias'),
    ]

    num_layers_1 = 5
    for idx in range(num_layers_1):
        ly_py_prefix = f'blocks.block.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.relative_position_bias_table', f'{ly_py_prefix}.attn.relative_position_bias_table'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping.extend(layer_mapping)
        
    num_layers_2 = 4
    for idx in range(num_layers_2):
        ly_py_prefix = f'upsample_blocks.0.block.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.relative_position_bias_table', f'{ly_py_prefix}.attn.relative_position_bias_table'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping.extend(layer_mapping)
        
    num_layers_3 = 2
    for idx in range(num_layers_3):
        ly_py_prefix = f'upsample_blocks.1.block.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.relative_position_bias_table', f'{ly_py_prefix}.attn.relative_position_bias_table'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping.extend(layer_mapping)

    head_mapping = [
        (f'{py_prefix}.deconv.0.weight', 'deconv.0.weight'),
        (f'{py_prefix}.deconv.0.bias', 'deconv.0.bias')
    ]
    mapping.extend(head_mapping)

    return mapping

def torch_to_paddle_mapping_dis():
    py_prefix = 'module'
    mapping_all = []
    
    mapping_dis = [
        (f'{py_prefix}.cls_token', 'cls_token'),
        (f'{py_prefix}.pos_embed_1', 'pos_embed_1'),
        (f'{py_prefix}.pos_embed_2', 'pos_embed_2'),
        (f'{py_prefix}.fRGB_1.weight', 'fRGB_1.weight'),
        (f'{py_prefix}.fRGB_1.bias', 'fRGB_1.bias'),
        (f'{py_prefix}.fRGB_2.weight', 'fRGB_2.weight'),
        (f'{py_prefix}.fRGB_2.bias', 'fRGB_2.bias'),
    ]

    num_layers_1 = 2
    for idx in range(num_layers_1):
        ly_py_prefix = f'blocks_1.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.relative_position_bias_table', f'{ly_py_prefix}.attn.relative_position_bias_table'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping_dis.extend(layer_mapping)
        
    num_layers_2 = 2
    for idx in range(num_layers_2):
        ly_py_prefix = f'blocks_2.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.relative_position_bias_table', f'{ly_py_prefix}.attn.relative_position_bias_table'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping_dis.extend(layer_mapping)
    
    num_layers_3 = 1
    for idx in range(num_layers_3):
        ly_py_prefix = f'last_block.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.qkv.weight', f'{ly_py_prefix}.attn.qkv.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.weight', f'{ly_py_prefix}.attn.proj.weight'),
            (f'{py_prefix}.{ly_py_prefix}.attn.proj.bias', f'{ly_py_prefix}.attn.proj.bias'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.weight', f'{ly_py_prefix}.norm2.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm2.norm.bias', f'{ly_py_prefix}.norm2.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.weight', f'{ly_py_prefix}.mlp.fc1.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc1.bias', f'{ly_py_prefix}.mlp.fc1.bias'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.weight', f'{ly_py_prefix}.mlp.fc2.weight'),
            (f'{py_prefix}.{ly_py_prefix}.mlp.fc2.bias', f'{ly_py_prefix}.mlp.fc2.bias'),
        ]
        mapping_dis.extend(layer_mapping)
        

    head_mapping = [
        (f'{py_prefix}.norm.norm.weight', 'norm.norm.weight'),
        (f'{py_prefix}.norm.norm.bias', 'norm.norm.bias'),
        (f'{py_prefix}.head.weight', 'head.weight'),
        (f'{py_prefix}.head.bias', 'head.bias'),
    ]
    mapping_dis.extend(head_mapping)
    

    return mapping_dis



def convert(torch_model, paddle_model, mapping):
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

        print("value.shape",str(value.shape)[1:-2])
        print("pd_params[pd_name].shape",str(pd_params[pd_name].shape)[1:-2])
        if str(value.shape)[1:-2] == str(pd_params[pd_name].shape)[1:-2]:
            pd_params[pd_name].set_value(value)
        else:
            pd_params[pd_name].set_value(value.T)

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
    if mapping == "gen":
        mapping = torch_to_paddle_mapping()
    else:
        mapping = torch_to_paddle_mapping_dis()

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            _set_value(th_name, pd_name)
        else: # weight & bias
            th_name_w = f'{th_name}.weight'
            pd_name_w = f'{pd_name}.weight'
            _set_value(th_name_w, pd_name_w)

            th_name_b = f'{th_name}.bias'
            pd_name_b = f'{pd_name}.bias'
            _set_value(th_name_b, pd_name_b)

    return paddle_model
    
def main():
    parser = argparse.ArgumentParser('transGAN')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
    args = parser.parse_args()

    # get default config
    config = get_config()
    # update config by arguments
    config = update_config(config, args)
    config.freeze()

    paddle.set_device('cpu')
    paddle_model_gen = Generator(args = config)
    paddle_model_dis = Discriminator(args = config)
    
    paddle_model_gen.eval()
    paddle_model_dis.eval()

    print_model_named_params(paddle_model_gen)
    print_model_named_buffers(paddle_model_gen)

    print_model_named_params(paddle_model_dis)
    print_model_named_buffers(paddle_model_dis)

    device = torch.device('cpu')
    torch_model_gen = eval('models_search.'+'ViT_custom_new'+'.Generator')(args=config)
    torch_model_gen = torch.nn.DataParallel(torch_model_gen.to("cuda:0"), device_ids=[0])

    torch_model_dis = eval('models_search.'+'ViT_custom_new'+'.Discriminator')(args=config)
    torch_model_dis = torch.nn.DataParallel(torch_model_dis.to("cuda:0"), device_ids=[0])

    checkpoint=torch.load("./cifar_checkpoint")
    torch_model_gen.load_state_dict(checkpoint['avg_gen_state_dict'])
    torch_model_dis.load_state_dict(checkpoint['dis_state_dict'])

    torch_model_gen = torch_model_gen.to(device)
    torch_model_gen.eval()
    torch_model_dis = torch_model_dis.to(device)
    torch_model_dis.eval()

    #convert weights
    paddle_model_gen = convert(torch_model_gen, paddle_model_gen, "gen")
    paddle_model_dis = convert(torch_model_dis, paddle_model_dis, "dis")

    # check correctness
    
    #save weights for paddle model
    model_path = os.path.join('./transgan_cifar10.pdparams')
    paddle.save({
          'gen_state_dict': paddle_model_gen.state_dict(),
          'dis_state_dict': paddle_model_dis.state_dict(),
            }, model_path)
    print('all done')


if __name__ == "__main__":
    main()
