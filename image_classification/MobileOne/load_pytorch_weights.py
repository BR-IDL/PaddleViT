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
from mobileone import build_mobileone as build_model
from mobileone import model_convert
from config import get_config
from pth_mobileone import mobileone as mobileone_pytorch
from pth_mobileone import reparameterize_model



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
    mapping = []
    th_prefix = f'stage0'
    pp_prefix = f'stages.0.0'
    layer_mapping = [
        (f'{th_prefix}.rbr_conv.0.conv', f'{pp_prefix}.dw_3x3_blocks.0.conv'),
        (f'{th_prefix}.rbr_conv.0.bn', f'{pp_prefix}.dw_3x3_blocks.0.norm'),
        (f'{th_prefix}.rbr_scale.conv', f'{pp_prefix}.dw_1x1.conv'),
        (f'{th_prefix}.rbr_scale.bn', f'{pp_prefix}.dw_1x1.norm'),
    ]
    mapping.extend(layer_mapping)
    
    for stage_idx in range(1, len(config.MODEL.NUM_BLOCKS)):
        br = config.MODEL.NUM_BRANCHES[stage_idx]
        for block_idx in range(config.MODEL.NUM_BLOCKS[stage_idx]):
            th_prefix = f'stage{stage_idx}.{block_idx * 2}'
            pp_prefix = f'stages.{stage_idx}.{block_idx}'
            for br_idx in range(br):
                layer_mapping = [
                    (f'{th_prefix}.rbr_conv.{br_idx}.conv', f'{pp_prefix}.dw_3x3_blocks.{br_idx}.conv'),
                    (f'{th_prefix}.rbr_conv.{br_idx}.bn', f'{pp_prefix}.dw_3x3_blocks.{br_idx}.norm'),
                ]
                mapping.extend(layer_mapping)
            layer_mapping = [
                (f'{th_prefix}.rbr_scale.conv', f'{pp_prefix}.dw_1x1.conv'),
                (f'{th_prefix}.rbr_scale.bn', f'{pp_prefix}.dw_1x1.norm'),
            ]
            mapping.extend(layer_mapping)
            if block_idx != 0: # stride == 1
                layer_mapping = [
                    (f'{th_prefix}.rbr_skip', f'{pp_prefix}.dw_skip'),
                ]
                mapping.extend(layer_mapping)


            th_prefix = f'stage{stage_idx}.{block_idx * 2 + 1}'
            for br_idx in range(br):
                layer_mapping = [
                    (f'{th_prefix}.rbr_conv.{br_idx}.conv', f'{pp_prefix}.pw_1x1_blocks.{br_idx}.conv'),
                    (f'{th_prefix}.rbr_conv.{br_idx}.bn', f'{pp_prefix}.pw_1x1_blocks.{br_idx}.norm'),
                ]
                mapping.extend(layer_mapping)
            layer_mapping = [
                (f'{th_prefix}.rbr_skip', f'{pp_prefix}.pw_skip'),
            ]
            mapping.extend(layer_mapping)


    # hard code for s4: stage 3 and stage 4 se blocks
    if config.MODEL.USE_SE:
        layer_mapping = [
            ('stage3.10.se.reduce', 'stages.3.5.dw_se.reduce'),
            ('stage3.10.se.expand', 'stages.3.5.dw_se.expand'),
            ('stage3.11.se.reduce', 'stages.3.5.pw_se.reduce'),
            ('stage3.11.se.expand', 'stages.3.5.pw_se.expand'),
            ('stage3.12.se.reduce', 'stages.3.6.dw_se.reduce'),
            ('stage3.12.se.expand', 'stages.3.6.dw_se.expand'),
            ('stage3.13.se.reduce', 'stages.3.6.pw_se.reduce'),
            ('stage3.13.se.expand', 'stages.3.6.pw_se.expand'),
            ('stage3.14.se.reduce', 'stages.3.7.dw_se.reduce'),
            ('stage3.14.se.expand', 'stages.3.7.dw_se.expand'),
            ('stage3.15.se.reduce', 'stages.3.7.pw_se.reduce'),
            ('stage3.15.se.expand', 'stages.3.7.pw_se.expand'),
            ('stage3.16.se.reduce', 'stages.3.8.dw_se.reduce'),
            ('stage3.16.se.expand', 'stages.3.8.dw_se.expand'),
            ('stage3.17.se.reduce', 'stages.3.8.pw_se.reduce'),
            ('stage3.17.se.expand', 'stages.3.8.pw_se.expand'),
            ('stage3.18.se.reduce', 'stages.3.9.dw_se.reduce'),
            ('stage3.18.se.expand', 'stages.3.9.dw_se.expand'),
            ('stage3.19.se.reduce', 'stages.3.9.pw_se.reduce'),
            ('stage3.19.se.expand', 'stages.3.9.pw_se.expand'),
            ('stage4.0.se.reduce', 'stages.4.0.dw_se.reduce'),
            ('stage4.0.se.expand', 'stages.4.0.dw_se.expand'),
            ('stage4.1.se.reduce', 'stages.4.0.pw_se.reduce'),
            ('stage4.1.se.expand', 'stages.4.0.pw_se.expand'),
        ]
        mapping.extend(layer_mapping)
        

    head_mapping = [
        ('linear', 'fc'),
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
    paddle.set_device('cpu')
    model_name_list = [
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
    ]

    for model_name in model_name_list:
        print(f'============= NOW: {model_name} =============')
        
        sz = 224 #int(model_name.split('_')[-1]) 
        print('Input Size: ', sz)

        config = get_config(f'./configs/mobileone_{model_name}.yaml')
        
        #config.defrost()
        #if '22k' in model_name and '1k' not in model_name:
        #    config.MODEL.NUM_CLASSES = 21841
        #if '73m' in model_name and '1k' not in model_name:
        #    config.MODEL.NUM_CLASSES = 730 #TODO: check class nums
        #config.freeze()

        paddle_model = build_model(config)

        paddle_model.eval()
        print_model_named_params(paddle_model)
        print_model_named_buffers(paddle_model)
        #print(paddle_model)

        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        print('+++++++++++++++++++++++++++++++++++')
        device = torch.device('cpu')
    
        inference_mode = False
        pth_name = f'mobileone_{model_name}.pth.tar' if inference_mode else f'mobileone_{model_name}_unfused.pth.tar'
        print(f'Load from {pth_name}')
        torch_model = mobileone_pytorch(num_classes=1000, inference_mode=inference_mode, variant=model_name)
        state_dict = torch.load(f'/Users/zhuyu05/Downloads/pth_weights_mobileone/{pth_name}', map_location=torch.device('cpu'))
        torch_model.load_state_dict(state_dict)
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
        out_paddle = paddle_model(x_paddle)

        out_torch = out_torch.data.cpu().numpy()
        out_paddle = out_paddle.cpu().numpy()

        print(out_torch.shape, out_paddle.shape)
        print(out_torch[0, 0:100])
        print('========================================================')
        print(out_paddle[0, 0:100])
        assert np.allclose(out_torch, out_paddle, atol = 1e-5)

        paddle_model_fused = model_convert(paddle_model, paddle.rand([2, 3, 224, 224]))

        # save weights for paddle model
        model_path = os.path.join(f'./{model_name}.pdparams')
        paddle.save(paddle_model.state_dict(), model_path)
        print(f'{model_name} done')

        model_path = os.path.join(f'./{model_name}_fused.pdparams')
        paddle.save(paddle_model_fused.state_dict(), model_path)
        print(f'{model_name} done')

    print('all done')


if __name__ == "__main__":
    main()
