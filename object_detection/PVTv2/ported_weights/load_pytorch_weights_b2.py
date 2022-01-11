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

#from misc import NestedTensor as ThNestedTensor
import os
import argparse
import numpy as np
import paddle
import torch
from config import get_config
from pvtv2_det import build_pvtv2_det
from model_utils import DropPath

#from pvt_det_pth.PVT.detection

#import timm
#from transformer import *
#from config import *
#from detr import build_detr
from utils import NestedTensor
from misc import NestedTensor as ThNestedTensor

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


config = get_config('./configs/pvtv2_b2.yaml')


def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)


def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)


def torch_to_paddle_mapping():
    map1 = torch_to_paddle_mapping_backbone()
    map2 = torch_to_paddle_mapping_neck()
    map3 = torch_to_paddle_mapping_head()
    map1.extend(map2)
    map1.extend(map3)
    return map1


def torch_to_paddle_mapping_neck():
    mapping = []
    for i in range(len(config.MODEL.TRANS.OUT_INDICES)):
        th_prefix = f'neck.lateral_convs.{i}.conv'
        pp_prefix = f'neck.fpn_lateral{i+2}.conv'
        mapping.append((th_prefix, pp_prefix))

        th_prefix = f'neck.fpn_convs.{i}.conv'
        pp_prefix = f'neck.fpn_output{i+2}.conv'
        mapping.append((th_prefix, pp_prefix))

    return mapping


def torch_to_paddle_mapping_head():
    mapping = [
            ('rpn_head.rpn_conv', 'rpnhead.conv'),
            ('rpn_head.rpn_cls', 'rpnhead.objectness_logits'),
            ('rpn_head.rpn_reg', 'rpnhead.anchor_deltas'),
            ('roi_head.bbox_head.fc_cls', 'roihead.predictor.cls_fc'),
            ('roi_head.bbox_head.fc_reg', 'roihead.predictor.reg_fc'),
            ('roi_head.bbox_head.shared_fcs.0', 'roihead.predictor.forward_net.linear0'),
            ('roi_head.bbox_head.shared_fcs.1', 'roihead.predictor.forward_net.linear1'),
            ]
    # Add mask head
    
    return mapping


def torch_to_paddle_mapping_backbone():
    mapping = []

    for embed_idx in range(1, 5):
        th_embed_prefix = f'backbone.patch_embed{embed_idx}'
        pp_embed_prefix = f'backbone.patch_embedding{embed_idx}'

        mapping.append((f'{th_embed_prefix}.proj',
                        f'{pp_embed_prefix}.patch_embed'))
        mapping.append((f'{th_embed_prefix}.norm',
                        f'{pp_embed_prefix}.norm'))

    for i in range(5):
        mapping.append((f'backbone.norm{i}',
                        f'backbone.norm{i}'))

    block_depth = config.MODEL.TRANS.STAGE_DEPTHS # [2, 2, 2, 2]

    for block_idx in range(1, len(block_depth) + 1):
        th_block_prefix = f'backbone.block{block_idx}'
        pp_block_prefix = f'backbone.block{block_idx}'

        for layer_idx in range(block_depth[block_idx-1]):
            th_prefix = f'{th_block_prefix}.{layer_idx}'
            pp_prefix = f'{pp_block_prefix}.{layer_idx}'
            layer_mapping = [
                (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                (f'{th_prefix}.attn.q', f'{pp_prefix}.attn.q'),
                (f'{th_prefix}.attn.kv', f'{pp_prefix}.attn.kv'),
                (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
                (f'{th_prefix}.attn.sr', f'{pp_prefix}.attn.sr'),
                (f'{th_prefix}.attn.norm', f'{pp_prefix}.attn.norm'),
                (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
                (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
                (f'{th_prefix}.mlp.dwconv.dwconv', f'{pp_prefix}.mlp.dwconv.dwconv'),
            ]
            mapping.extend(layer_mapping)
    return mapping


def convert_from_torch_state_dict(torch_model_state_dict, paddle_model):
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

    # 1. get paddle and torch model parameters
    pd_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, buff in paddle_model.named_buffers():
        pd_params[name] = buff

    th_params = torch_model_state_dict

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            _set_value(th_name, pd_name)
        else: # weight & bias
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)
        
            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def get_nested_tensors():
    with open('./t.npy', 'rb') as infile:
        t = np.load(infile)
        m = np.load(infile)
        gts = np.load(infile, allow_pickle=True).item()

    #print(t.shape)
    #print(m.shape)

    tt = torch.Tensor(t)
    mm = torch.Tensor(m)
    th_in = th_utils.NestedTensor(tt, mm)

    ttt = paddle.to_tensor(t)
    mmm = paddle.to_tensor(m)
    pp_in = NestedTensor(ttt, mmm)

    #print(th_in, th_in.tensors.shape)
    #print(pp_in, pp_in.tensors.shape)

    targets = {}
    for key, gt in gts.items():
        targets[key] = []
        for val in gt:
            targets[key].append(paddle.to_tensor(val))
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

    paddle.set_device('cpu')

    #th_in, th_target, pp_in, pp_target = get_nested_tensors()
    
    paddle_model  = build_pvtv2_det(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)
    print('------------paddle model finish ----------------------')


    #device = torch.device('cpu')
    #torch_model = 
    #torch_model = torch_model.to(device)
    #torch_model.eval()

    #print_model_named_params(torch_model)
    #print_model_named_buffers(torch_model)
    #print('----------torch model finish------------------------')
     
    torch_state_dict = torch.load('./pth_weights/mask_rcnn_pvt_v2_b2_fpn_1x_coco.pth')
    # dict_keys(['meta', 'state_dict', 'optimizer'])
    for key, val in torch_state_dict['state_dict'].items():
        print(key, val.shape)
    print('----------torch model finish------------------------')
    torch_model_state_dict = torch_state_dict['state_dict']

    # convert weights
    paddle_model = convert_from_torch_state_dict(torch_model_state_dict, paddle_model)


    # check correctness
    #th_in, th_target, pp_in, pp_target = get_nested_tensors()
    #th_in, th_target, pp_in, pp_target = get_nested_tensors_random()
    #x = np.random.randn(1, 3, 224, 224).astype('float32')
    #x_paddle = paddle.to_tensor(x)
    #x_torch = torch.Tensor(x).to(device)



    #print(pp_in.tensors)
    #print(pp_in.mask)
    #print('-------- pp in finish ------------------')
    

    #print(th_in.tensors, th_in.tensors.shape)
    #print(th_in.mask, th_in.mask.shape)
    #print('-------- th in finish ------------------')

    # save weights for paddle model
    model_path = os.path.join('./pvtv2_b2_maskrcnn.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)
   


   # pp_in, th_in, pp_gt = get_nested_tensors()
   # print('pp_in: ', pp_in.tensors.shape)

   #  out_paddle = paddle_model(pp_in, pp_gt)
   #  print('paddle_out = ', out_paddle)




    #loss = paddle_criterion(out_paddle, pp_gt)
    #print('=============== loss =============')
    #for key, val in loss.items():
    #    print(key, val.cpu().numpy())

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
