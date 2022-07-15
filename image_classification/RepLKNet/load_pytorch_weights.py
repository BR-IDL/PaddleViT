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
from replknet import build_replknet as build_model
from config import get_config
from replknet_pth import create_RepLKNet31B
from replknet_pth import create_RepLKNet31L
from replknet_pth import create_RepLKNetXL


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
    for stem_idx in range(4):
        th_prefix = f'stem.{stem_idx}'
        pp_prefix = f'stem.{stem_idx}'
        layer_mapping = [
            (f'{th_prefix}.conv', f'{pp_prefix}.conv'),
            (f'{th_prefix}.bn', f'{pp_prefix}.norm'),
        ]
        mapping.extend(layer_mapping)
    
    for stage_idx in range(len(config.MODEL.LAYERS)):
        for block_idx in range(config.MODEL.LAYERS[stage_idx]):
            block_idx = block_idx * 2
            th_prefix = f'stages.{stage_idx}.blocks'
            pp_prefix = f'stages.{stage_idx}.blocks'
            layer_mapping = [
                (f'{th_prefix}.{block_idx}.pw1.conv', f'{pp_prefix}.{block_idx}.pw_conv_1.conv'),
                (f'{th_prefix}.{block_idx}.pw1.bn', f'{pp_prefix}.{block_idx}.pw_conv_1.norm'),
                (f'{th_prefix}.{block_idx}.pw2.conv', f'{pp_prefix}.{block_idx}.pw_conv_2.conv'),
                (f'{th_prefix}.{block_idx}.pw2.bn', f'{pp_prefix}.{block_idx}.pw_conv_2.norm'),
                (f'{th_prefix}.{block_idx}.large_kernel.lkb_origin.conv', f'{pp_prefix}.{block_idx}.large_kernel.large_kernel_origin.conv'),
                (f'{th_prefix}.{block_idx}.large_kernel.lkb_origin.bn', f'{pp_prefix}.{block_idx}.large_kernel.large_kernel_origin.norm'),
                (f'{th_prefix}.{block_idx}.large_kernel.small_conv.conv', f'{pp_prefix}.{block_idx}.large_kernel.small_kernel_conv.conv'),
                (f'{th_prefix}.{block_idx}.large_kernel.small_conv.bn', f'{pp_prefix}.{block_idx}.large_kernel.small_kernel_conv.norm'),
                (f'{th_prefix}.{block_idx}.prelkb_bn', f'{pp_prefix}.{block_idx}.pre_large_kernel_bn'),
                (f'{th_prefix}.{block_idx + 1}.preffn_bn', f'{pp_prefix}.{block_idx + 1}.pre_ffn_bn'),
                (f'{th_prefix}.{block_idx + 1}.pw1.conv', f'{pp_prefix}.{block_idx + 1}.pw_conv_1.conv'),
                (f'{th_prefix}.{block_idx + 1}.pw1.bn', f'{pp_prefix}.{block_idx + 1}.pw_conv_1.norm'),
                (f'{th_prefix}.{block_idx + 1}.pw2.conv', f'{pp_prefix}.{block_idx + 1}.pw_conv_2.conv'),
                (f'{th_prefix}.{block_idx + 1}.pw2.bn', f'{pp_prefix}.{block_idx + 1}.pw_conv_2.norm'),
            ]
            mapping.extend(layer_mapping)

            if config.MODEL.SMALL_KERNEL is not None:
                layer_mapping = [
                    (f'{th_prefix}.{block_idx}.large_kernel.small_conv.conv', f'{pp_prefix}.{block_idx}.large_kernel.small_kernel_conv.conv'),
                    (f'{th_prefix}.{block_idx}.large_kernel.small_conv.bn', f'{pp_prefix}.{block_idx}.large_kernel.small_kernel_conv.norm'),
                ]
                mapping.extend(layer_mapping)

    for trans_idx in range(len(config.MODEL.LAYERS) - 1):
        th_prefix = f'transitions.{trans_idx}'
        pp_prefix = f'transitions.{trans_idx}'
        layer_mapping = [
            (f'{th_prefix}.0.conv', f'{pp_prefix}.0.conv'),
            (f'{th_prefix}.0.bn', f'{pp_prefix}.0.norm'),
            (f'{th_prefix}.1.conv', f'{pp_prefix}.1.conv'),
            (f'{th_prefix}.1.bn', f'{pp_prefix}.1.norm'),
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
        #"replknet_31b_1k_224",
        #"replknet_31b_1k_384",
        #"replknet_31b_22k_224",
        #"replknet_31b_22k_to_1k_224",
        #"replknet_31b_22k_to_1k_384",
        #"replknet_31l_22k_224",
        #"replknet_31l_22k_to_1k_384",
        "replknet_xl_73m_to_1k_320",
        #"replknet_xl_73m_320",
    ]

    for model_name in model_name_list:
        print(f'============= NOW: {model_name} =============')
        
        sz = int(model_name.split('_')[-1]) 
        print('Input Size: ', sz)
        
        if '31b' in model_name:
            config = get_config(f'./configs/replknet_31b.yaml')
        elif '31l' in model_name:
            config = get_config(f'./configs/replknet_31l.yaml')
        elif 'xl' in model_name:
            config = get_config(f'./configs/replknet_xl.yaml')

        config.defrost()
        if '22k' in model_name and '1k' not in model_name:
            config.MODEL.NUM_CLASSES = 21841
        #if '73m' in model_name and '1k' not in model_name:
        #    config.MODEL.NUM_CLASSES = 730 #TODO: check class nums
        config.freeze()


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
    
        pth_name = 'RepLKNet-'
        if '31b' in model_name:
            create_model = create_RepLKNet31B
            pth_name += '31B'
        elif '31l' in model_name:
            create_model = create_RepLKNet31L
            pth_name += '31L'
        elif 'xl' in model_name:
            create_model = create_RepLKNetXL
            pth_name += 'XL'

        if '1k' in model_name:
            num_classes = 1000
            if '22k' in model_name:
                pth_name += f'_ImageNet-22K-to-1K_{sz}.pth'
            elif '73m' in model_name:
                pth_name += f'_MegData73M_ImageNet1K.pth'
            else:
                pth_name += f'_ImageNet-1K_{sz}.pth'

        if '22k' in model_name and '1k' not in model_name:
            num_classes = 21841
            pth_name += '_ImageNet-22K.pth'
        #if '73m' in model_name and '1k' not in model_name:
        #    num_classes = 0 
        #    pth_name += '_MegData73M-pretrain.pth'

        print(f'Load from {pth_name}')
        torch_model = create_model(num_classes=num_classes, small_kernel_merged=False, use_checkpoint=False)
        state_dict = torch.load(f'../../../../replknet_pth_weights/{pth_name}')
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
        assert np.allclose(out_torch, out_paddle, atol = 1e-4)

        # save weights for paddle model
        model_path = os.path.join(f'./{model_name}.pdparams')
        paddle.save(paddle_model.state_dict(), model_path)
        print(f'{model_name} done')
    print('all done')


if __name__ == "__main__":
    main()
