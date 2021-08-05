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

import sys
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import paddle
import TransGAN.models_search as models_search
from models.ViT_custom import Generator
from models.ViT_custom_scale2 import Discriminator
from config import get_config, update_config
import matplotlib.pyplot as plt

sys.path.append("../TransGAN")
sys.path.append("..")

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

    num_layers_1 = 3
    for idx in range(num_layers_1):
        ly_py_prefix = f'blocks_1.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.noise_strength_1', f'{ly_py_prefix}.attn.noise_strength_1'),
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

    num_layers_2 = 3
    for idx in range(num_layers_2):
        ly_py_prefix = f'blocks_2.{idx}'
        layer_mapping = [
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.weight', f'{ly_py_prefix}.norm1.norm.weight'),
            (f'{py_prefix}.{ly_py_prefix}.norm1.norm.bias', f'{ly_py_prefix}.norm1.norm.bias'),
            (f'{py_prefix}.{ly_py_prefix}.attn.noise_strength_1', f'{ly_py_prefix}.attn.noise_strength_1'),
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
            (f'{py_prefix}.{ly_py_prefix}.attn.noise_strength_1', f'{ly_py_prefix}.attn.noise_strength_1'),
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
        if th_name.find("attn.proj.weight") != -1 and th_shape == pd_shape: # prevent shape[1]==shape[0]
            value = value.transpose((1, 0))
        if len(value.shape) == 2:
            if not no_transpose:
                value = value.transpose((1, 0))
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

    parser = argparse.ArgumentParser()
    args_torch = parser.parse_args()
    with open('../TransGAN/commandline_args.txt', 'r') as f:
        args_torch.__dict__ = json.load(f)

    paddle.set_device('cpu')
    paddle_model_gen = Generator(args=config)
    paddle_model_dis = Discriminator(args=config)
    
    paddle_model_gen.eval()
    paddle_model_dis.eval()

    print_model_named_params(paddle_model_gen)
    print_model_named_buffers(paddle_model_gen)

    print_model_named_params(paddle_model_dis)
    print_model_named_buffers(paddle_model_dis)

    device = torch.device('cpu')
    torch_model_gen = eval('models_search.'+'ViT_custom_new'+'.Generator')(args=args_torch)
    torch_model_gen = torch.nn.DataParallel(torch_model_gen.to("cuda:0"), device_ids=[0])

    torch_model_dis = eval('models_search.'+'ViT_custom_scale2'+'.Discriminator')(args=args_torch)
    torch_model_dis = torch.nn.DataParallel(torch_model_dis.to("cuda:0"), device_ids=[0])

    print_model_named_params(torch_model_gen)
    print_model_named_buffers(torch_model_gen)

    print_model_named_params(torch_model_dis)
    print_model_named_buffers(torch_model_dis)

    checkpoint = torch.load("../cifar_checkpoint")
    torch_model_gen.load_state_dict(checkpoint['avg_gen_state_dict'])
    torch_model_dis.load_state_dict(checkpoint['dis_state_dict'])

    torch_model_gen = torch_model_gen.to(device)
    torch_model_gen.eval()
    torch_model_dis = torch_model_dis.to(device)
    torch_model_dis.eval()

    # convert weights
    paddle_model_gen = convert(torch_model_gen, paddle_model_gen, "gen")
    paddle_model_dis = convert(torch_model_dis, paddle_model_dis, "dis")

    # check correctness
    x = np.random.normal(0, 1, (args_torch.eval_batch_size, args_torch.latent_dim))
    z_paddle = paddle.to_tensor(x, dtype="float32")
    z_torch = torch.cuda.FloatTensor(x)
    epoch = 0
    device_cor = torch.device('cuda')
    torch_model_gen = torch_model_gen.to(device_cor)
    torch_model_dis = torch_model_dis.to(device_cor)
    gen_imgs_torch = torch_model_gen(z_torch, epoch)
    fake_validity_torch = torch_model_dis(gen_imgs_torch)
    gen_imgs_torch = gen_imgs_torch.mul_(127.5).add_(127.5).clamp_(0.0, 255.0)
    gen_imgs_torch = gen_imgs_torch.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    plt.figure()
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(gen_imgs_torch[i-1])
        plt.xticks([])
        plt.yticks([])
    plt.draw()
    plt.savefig(str("test_torch") + '.png')
    print("gen_img_torch", gen_imgs_torch.flatten()[:10])
    print("fake_validity_torch", fake_validity_torch.flatten()[:5])

    model_state = paddle.load('./transgan_cifar10.pdparams')
    paddle_model_gen.set_dict(model_state['gen_state_dict'])
    paddle_model_dis.set_dict(model_state['dis_state_dict'])
    gen_imgs_paddle = paddle_model_gen(z_paddle, epoch)
    fake_validity_paddle = paddle_model_dis(gen_imgs_paddle)
    gen_imgs_paddle = paddle.add(paddle.multiply(gen_imgs_paddle, paddle.to_tensor(127.5)), paddle.to_tensor(127.5))
    gen_imgs_paddle = paddle.clip(gen_imgs_paddle.transpose((0, 2, 3, 1)), min=0.0, max=255.0).astype('uint8').cpu().numpy()
    plt.figure()
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        plt.imshow(gen_imgs_paddle[i-1])
        plt.xticks([])
        plt.yticks([])
    plt.draw()
    plt.savefig(str("test_paddle") + '.png')
    print("gen_imgs_paddle", gen_imgs_paddle.flatten()[:10])
    print("fake_validity_paddle", fake_validity_paddle.flatten()[:5])
    
    #save weights for paddle model
    model_path = os.path.join('./transgan_cifar10.pdparams')
    paddle.save({
        'gen_state_dict': paddle_model_gen.state_dict(),
        'dis_state_dict': paddle_model_dis.state_dict(),
    }, model_path)
    print('all done')

if __name__ == "__main__":
    main()
