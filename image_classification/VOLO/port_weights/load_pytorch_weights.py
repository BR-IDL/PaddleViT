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
from config import *
from volo import *
from pytorch.volo.models.volo import volo_d1, volo_d2, volo_d3, volo_d4, volo_d5
from pytorch.volo.utils import load_pretrained_weights


names = [
    ('volo_d5_448', 'd5_448_87.0', volo_d5),
    ('volo_d4_448', 'd4_448_86.79', volo_d4),
    ('volo_d4_224', 'd4_224_85.7', volo_d4),
    ('volo_d3_448', 'd3_448_86.3', volo_d3),
    ('volo_d3_224', 'd3_224_85.4', volo_d3),
    ('volo_d2_384', 'd2_384_86.0', volo_d2),
    ('volo_d2_224', 'd2_224_85.2', volo_d2),
    ('volo_d1_384', 'd1_384_85.2', volo_d1),
    ('volo_d1_224', 'd1_224_84.2', volo_d1),
    ]
idx = 8
gmodel_name = names[idx][0]
gmodel_path = names[idx][1]
sz = int(gmodel_name[-3::])
model_type=names[idx][2]


config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg', type=str, default=f'./configs/{gmodel_name}.yaml')
#parser.add_argument('-cfg', type=str, default='./configs/volo_d5_224.yaml')
parser.add_argument('-dataset', type=str, default="imagenet2012")
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
        ('cls_token', 'cls_token'),
        ('pos_embed', 'pos_embed'),
        ('patch_embed.proj', 'patch_embed.proj'),
    ]
    
    # patch embedding:
    th_prefix = 'patch_embed.conv'
    pp_prefix = 'patch_embed.stem'
    layer_mapping = [
        (f'{th_prefix}.0.weight', f'{pp_prefix}.0.weight'),#conv
        (f'{th_prefix}.1.weight', f'{pp_prefix}.1.weight'),#bn
        (f'{th_prefix}.1.bias', f'{pp_prefix}.1.bias'),#bn
        (f'{th_prefix}.1.running_mean', f'{pp_prefix}.1._mean'),#bn
        (f'{th_prefix}.1.running_var', f'{pp_prefix}.1._variance'),#bn
        (f'{th_prefix}.3.weight', f'{pp_prefix}.3.weight'),#conv
        (f'{th_prefix}.4.weight', f'{pp_prefix}.4.weight'),#bn
        (f'{th_prefix}.4.bias', f'{pp_prefix}.4.bias'),#bn
        (f'{th_prefix}.4.running_mean', f'{pp_prefix}.4._mean'),#bn
        (f'{th_prefix}.4.running_var', f'{pp_prefix}.4._variance'),#bn
        (f'{th_prefix}.6.weight', f'{pp_prefix}.6.weight'),#conv
        (f'{th_prefix}.7.weight', f'{pp_prefix}.7.weight'),#bn
        (f'{th_prefix}.7.bias', f'{pp_prefix}.7.bias'),#bn
        (f'{th_prefix}.7.running_mean', f'{pp_prefix}.7._mean'),#bn
        (f'{th_prefix}.7.running_var', f'{pp_prefix}.7._variance'),#bn
    ]
    mapping.extend(layer_mapping)

    # models
    for idx, stage_idx in enumerate([0, 2, 3, 4]):
        for layer_idx in range(config.MODEL.TRANS.LAYERS[idx]):
            pp_prefix = f'model.{stage_idx}.{layer_idx}'
            th_prefix = f'network.{stage_idx}.{layer_idx}'

            if config.MODEL.TRANS.OUTLOOK_ATTENTION[idx]:
                layer_mapping = [
                    (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                    (f'{th_prefix}.attn.v.weight', f'{pp_prefix}.attn.v.weight'),
                    (f'{th_prefix}.attn.attn', f'{pp_prefix}.attn.attn'),
                    (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
                    (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                    (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
                    (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
                ]
            else:
                layer_mapping = [
                    (f'{th_prefix}.norm1', f'{pp_prefix}.norm1'),
                    (f'{th_prefix}.attn.qkv.weight', f'{pp_prefix}.attn.qkv.weight'),
                    (f'{th_prefix}.attn.proj', f'{pp_prefix}.attn.proj'),
                    (f'{th_prefix}.norm2', f'{pp_prefix}.norm2'),
                    (f'{th_prefix}.mlp.fc1', f'{pp_prefix}.mlp.fc1'),
                    (f'{th_prefix}.mlp.fc2', f'{pp_prefix}.mlp.fc2'),
                ]
            mapping.extend(layer_mapping)

    layer_mapping = [
        ('network.1.proj', 'model.1.proj'),
    ]
    mapping.extend(layer_mapping)
    # Post layers
    pp_prefix = f'post_model'
    th_prefix = f'post_network'
    for idx in range(2):
        layer_mapping = [
            (f'{th_prefix}.{idx}.norm1', f'{pp_prefix}.{idx}.norm1'),
            (f'{th_prefix}.{idx}.attn.kv.weight', f'{pp_prefix}.{idx}.attn.kv.weight'),
            (f'{th_prefix}.{idx}.attn.q.weight', f'{pp_prefix}.{idx}.attn.q.weight'),
            (f'{th_prefix}.{idx}.attn.proj', f'{pp_prefix}.{idx}.attn.proj'),
            (f'{th_prefix}.{idx}.norm2', f'{pp_prefix}.{idx}.norm2'),
            (f'{th_prefix}.{idx}.mlp.fc1', f'{pp_prefix}.{idx}.mlp.fc1'),
            (f'{th_prefix}.{idx}.mlp.fc2', f'{pp_prefix}.{idx}.mlp.fc2'),
        ]
        mapping.extend(layer_mapping)
    # Head layers
    head_mapping = [
        ('aux_head', 'aux_head'),
        ('norm', 'norm'),
        ('head', 'head')
    ]
    mapping.extend(head_mapping)

    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].cpu().data.numpy()
        if len(value.shape) == 2:
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

    paddle.set_device('cpu')
    paddle_model = build_volo(config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    device = torch.device('cpu')
    torch_model = model_type(img_size=config.DATA.IMAGE_SIZE)


    #torch_model = volo_d5(img_size=config.DATA.IMAGE_SIZE) 
    load_pretrained_weights(torch_model, f'./pytorch/volo/{gmodel_path}.pth.tar',
    #load_pretrained_weights(torch_model, './pytorch/volo/d5_224_86.10.pth.tar',
        use_ema=False, strict=False, num_classes=1000)
    torch_model = torch_model.to(device)
    torch_model.eval()

    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, sz, sz).astype('float32')
    #x = np.random.randn(2, 3, 224, 224).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torch_model(x_torch)
    print('========================================================')
    print('========================================================')
    print('========================================================')
    print('========================================================')
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[1, 0:100])
    print(out_paddle[1, 0:100])
    assert np.allclose(out_torch[0], out_paddle[0], atol = 1e-3)
    print('===== out 0 equal OK')
    assert np.allclose(out_torch[1], out_paddle[1], atol = 1e-3)
    print('===== out 1 equal OK')
    
    # save weights for paddle model
    print('===== saving .pdparams')
    model_path = os.path.join(f'./{gmodel_path}.pdparams')
    #model_path = os.path.join('./d5_512_87.07.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)
    print('all done')


if __name__ == "__main__":
    main()
