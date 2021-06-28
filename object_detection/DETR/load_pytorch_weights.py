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

import sys
sys.path.append('/root/.cache/torch/hub/facebookresearch_detr_master/util/')

from misc import NestedTensor as ThNestedTensor
import os
import argparse
import numpy as np
import paddle
import torch
#import timm
#from transformer import *
#from config import *
from detr import build_detr
from utils import NestedTensor

import misc as th_utils
#config = get_config()
#parser = argparse.ArgumentParser('')
#parser.add_argument('-cfg', type=str, default='./configs/vit_large_patch16_224.yaml')
##parser.add_argument('-dataset', type=str, default="imagenet2012")
#parser.add_argument('-dataset', type=str, default="cifar10")
#parser.add_argument('-batch_size', type=int, default=4)
#parser.add_argument('-image_size', type=int, default=224)
#parser.add_argument('-data_path', type=str, default='/dataset/imagenet/')
#parser.add_argument('-eval', action="store_true")
#parser.add_argument('-pretrained', type=str, default=None)
#args = parser.parse_args()
#
#config = get_config()
#config = update_config(config, args)
#print(config)
#
#
def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)


def torch_to_paddle_mapping():
    map1 = torch_to_paddle_mapping_backbone()
    map2 = torch_to_paddle_mapping_transformer()
    map3 = torch_to_paddle_mapping_bn_from_buffer()
    map1.extend(map2)
    map1.extend(map3)
    return map1

def torch_to_paddle_mapping_bn_from_buffer():
    mapping = [('backbone.0.body.bn1','backbone.0.body.bn1')]

    block_depth = [3, 4, 6, 3]
    for block_idx in range(1,5):
        th_block_prefix = f'backbone.0.body.layer{block_idx}'
        pp_block_prefix = f'backbone.0.body.layer{block_idx}'
        mapping.append((f'{th_block_prefix}.0.downsample.1',
                        f'{pp_block_prefix}.0.downsample.1'))

        for layer_idx in range(block_depth[block_idx-1]):
            th_prefix = f'{th_block_prefix}.{layer_idx}'
            pp_prefix = f'{pp_block_prefix}.{layer_idx}'
            layer_mapping = [
                (f'{th_prefix}.bn1', f'{pp_prefix}.bn1'),
                (f'{th_prefix}.bn2', f'{pp_prefix}.bn2'),
                (f'{th_prefix}.bn3', f'{pp_prefix}.bn3'),
            ]
            mapping.extend(layer_mapping)
    return mapping

def torch_to_paddle_mapping_backbone():
    mapping = [('backbone.0.body.conv1','backbone.0.body.conv1')]

    block_depth = [3, 4, 6, 3]
    for block_idx in range(1,5):
        th_block_prefix = f'backbone.0.body.layer{block_idx}'
        pp_block_prefix = f'backbone.0.body.layer{block_idx}'
        mapping.append((f'{th_block_prefix}.0.downsample.0',
                        f'{pp_block_prefix}.0.downsample.0'))

        for layer_idx in range(block_depth[block_idx-1]):
            th_prefix = f'{th_block_prefix}.{layer_idx}'
            pp_prefix = f'{pp_block_prefix}.{layer_idx}'
            layer_mapping = [
                (f'{th_prefix}.conv1', f'{pp_prefix}.conv1'),
                (f'{th_prefix}.conv2', f'{pp_prefix}.conv2'),
                (f'{th_prefix}.conv3', f'{pp_prefix}.conv3'),
            ]
            mapping.extend(layer_mapping)
    return mapping


def torch_to_paddle_mapping_transformer():
    mapping = [
        ('class_embed', 'class_embed'),
        ('query_embed', 'query_embed'),
        ('input_proj', 'input_proj'),
        ('bbox_embed.layers.0', 'bbox_embed.layers.0'),
        ('bbox_embed.layers.1', 'bbox_embed.layers.1'),
        ('bbox_embed.layers.2', 'bbox_embed.layers.2'),
        ('transformer.decoder.norm', 'transformer.decoder.norm'),
    ]

    num_layers = 6
    for idx in range(num_layers):
        for module in ['encoder', 'decoder']:
            pp_prefix = f'transformer.{module}.layers.{idx}'
            th_prefix = f'transformer.{module}.layers.{idx}'
            layer_mapping = [
                (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                (f'{th_prefix}.norm3', f'{pp_prefix}.norm3'),
                (f'{th_prefix}.linear1', f'{pp_prefix}.mlp.linear1'), 
                (f'{th_prefix}.linear2', f'{pp_prefix}.mlp.linear2'), 
                (f'{th_prefix}.self_attn.in_proj_weight', f'{pp_prefix}.self_attn'),
                (f'{th_prefix}.self_attn.in_proj_bias', f'{pp_prefix}.self_attn'),
                (f'{th_prefix}.self_attn.out_proj', f'{pp_prefix}.self_attn.fc'),
                (f'{th_prefix}.multihead_attn.in_proj_weight', f'{pp_prefix}.dec_enc_attn'),
                (f'{th_prefix}.multihead_attn.in_proj_bias', f'{pp_prefix}.dec_enc_attn'),
                (f'{th_prefix}.multihead_attn.out_proj', f'{pp_prefix}.dec_enc_attn.fc'),
            ]
            mapping.extend(layer_mapping)
    return mapping



def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'***SET*** {th_name} {th_shape} ***TO*** {pd_name} {pd_shape}')
        if isinstance(th_params[th_name], torch.nn.parameter.Parameter):
            value = th_params[th_name].data.numpy()
        else:
            value = th_params[th_name].numpy()
        if len(value.shape) == 2 and transpose:
            value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)

    def _set_value_attn(th_name, pd_name):
        th_shape = th_params[th_name].shape
        print(f'***SET*** {th_name} {th_shape} ***TO*** {pd_name}')
        if 'weight' in th_name:
            value = th_params[th_name].data.transpose(1, 0)
            value = value.chunk(3, axis=-1)
            q,k,v = value[0].numpy(), value[1].numpy(), value[2].numpy()
            #q = q.transpose((1,0))
            #k = k.transpose((1,0))
            #v = v.transpose((1,0))
            pd_params[f'{pd_name}.q.weight'].set_value(q)
            pd_params[f'{pd_name}.k.weight'].set_value(k)
            pd_params[f'{pd_name}.v.weight'].set_value(v)
        elif 'bias' in th_name:
            value = th_params[th_name].data
            #print('00000000000000000000000000000000')
            #print(value.shape)
            #print(value)
            value = value.chunk(3, axis=-1)
            q,k,v = value[0].numpy(), value[1].numpy(), value[2].numpy()
            #print('00000 q_b 00000')
            #print(q)
            #print('00000 k_b 00000')
            #print(k)
            #print('00000 v_b 00000')
            #print(v)
            pd_params[f'{pd_name}.q.bias'].set_value(q)
            pd_params[f'{pd_name}.k.bias'].set_value(k)
            pd_params[f'{pd_name}.v.bias'].set_value(v)


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
            if 'self_attn' in th_name or 'multihead_attn' in th_name:
                _set_value_attn(th_name, pd_name)
            else:
                _set_value(th_name, pd_name)
        else: # weight & bias
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                if th_name_w == 'query_embed.weight':
                    _set_value(th_name_w, pd_name_w, transpose=False)
                else:
                    _set_value(th_name_w, pd_name_w)
        
            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_mean' in th_params.keys():
                th_name_mean = f'{th_name}.running_mean'
                pd_name_mean = f'{pd_name}._mean'
                _set_value(th_name_mean, pd_name_mean)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_mean = f'{th_name}.running_var'
                pd_name_mean = f'{pd_name}._variance'
                _set_value(th_name_mean, pd_name_mean)

    return paddle_model

    
def get_nested_tensors():
    with open('./t.npy', 'rb') as infile:
        t = np.load(infile)
        m = np.load(infile)
        gts = np.load(infile, allow_pickle=True)

    print(t.shape)
    print(m.shape)

    tt = torch.Tensor(t)
    mm = torch.Tensor(m)
    th_in = th_utils.NestedTensor(tt, mm)

    ttt = paddle.to_tensor(t)
    mmm = paddle.to_tensor(m)
    pp_in = NestedTensor(ttt, mmm)

    print(th_in, th_in.tensors.shape)
    print(pp_in, pp_in.tensors.shape)

    targets = []
    for gt in gts:
        target = dict()
        for key, val in gt.items():
            target[key] = paddle.to_tensor(val)
        targets.append(target)
    targets = tuple(targets)
    pp_gt = targets


    return pp_in, th_in, pp_gt




#def get_nested_tensors():
#    samples = paddle.load(path='./batch_samples_01.pdtensor')
#    pp_in = NestedTensor(samples['tensors'], samples['mask'])
#    pp_target = paddle.load(path='./batch_targets_01.pdtensor')
#
#    samples_tensor = samples['tensors'].cpu().numpy() 
#    samples_mask = samples['mask'].cpu().numpy() 
#    th_tensor = torch.Tensor(samples_tensor)
#    th_mask = torch.Tensor(samples_mask)
#    th_in = ThNestedTensor(th_tensor, th_mask)
#    th_target = []
#    for item in pp_target:
#        sample_gt = dict()
#        for key, val in item.items():
#            th_tensor = torch.Tensor(val.cpu().numpy())
#            sample_gt[key] = th_tensor
#        th_target.append(sample_gt)
#
#    return th_in, th_target, pp_in, pp_target


def get_nested_tensors_random():
    x = np.random.randn(1, 3, 224, 224).astype('float32')
    mask = np.ones([1, 224, 224])

    pp_x = paddle.to_tensor(x)
    pp_mask = paddle.to_tensor(mask)
    pp_in = NestedTensor(pp_x, pp_mask)
    th_tensor = torch.Tensor(x)
    th_mask = torch.Tensor(mask)
    th_in = ThNestedTensor(th_tensor, th_mask)
    th_target = []
    pp_target = []

    return th_in, th_target, pp_in, pp_target


def main():

    paddle.set_device('gpu')

    #th_in, th_target, pp_in, pp_target = get_nested_tensors()
    
    paddle_model, paddle_criterion, paddle_postprocessors = build_detr()
    paddle_model.eval()

    #print_model_named_params(paddle_model)
    #print_model_named_buffers(paddle_model)
    print('------------paddle model finish ----------------------')

    device = torch.device('cpu')
    torch_model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    torch_model = torch_model.to(device)
    torch_model.eval()

    #print_model_named_params(torch_model)
    #print_model_named_buffers(torch_model)
    print('----------torch model finish------------------------')
     


    # convert weights
    #paddle_model = convert(torch_model, paddle_model)

    model_dict = paddle.load('./detr_resnet50.pdparams')
    paddle_model.set_dict(model_dict)


    # check correctness
    #th_in, th_target, pp_in, pp_target = get_nested_tensors()
    #th_in, th_target, pp_in, pp_target = get_nested_tensors_random()
    #x = np.random.randn(1, 3, 224, 224).astype('float32')
    #x_paddle = paddle.to_tensor(x)
    #x_torch = torch.Tensor(x).to(device)


    pp_in, th_in, pp_gt = get_nested_tensors()

    #print(pp_in.tensors)
    #print(pp_in.mask)
    #print('-------- pp in finish ------------------')
    

    #print(th_in.tensors, th_in.tensors.shape)
    #print(th_in.mask, th_in.mask.shape)
    #print('-------- th in finish ------------------')
    

    out_paddle = paddle_model(pp_in)
    loss = paddle_criterion(out_paddle, pp_gt)
    print('=============== loss =============')
    for key, val in loss.items():
        print(key, val.cpu().numpy())

    #print(out_paddle['pred_logits'], out_paddle['pred_logits'].shape)
    #print(out_paddle['pred_boxes'], out_paddle['pred_boxes'].shape)
    #print('---------- paddle out finish ------------------------')

    #out_torch = torch_model(th_in)
    #print(out_torch['pred_logits'], out_torch['pred_logits'].shape)
    #print(out_torch['pred_boxes'], out_torch['pred_boxes'].shape)
    #print('---------- torch out finish ------------------------')

    #out_torch = out_torch.data.cpu().numpy()
    #out_paddle = out_paddle.cpu().numpy()

    #print(out_torch.shape, out_paddle.shape)
    #print(out_torch[0:100])
    #print(out_paddle[0:100])
    #assert np.allclose(out_torch, out_paddle, atol = 1e-5)
#    
    # save weights for paddle model
    #model_path = os.path.join('./detr_resnet50.pdparams')
    #paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
