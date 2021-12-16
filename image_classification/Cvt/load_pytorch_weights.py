import os
import numpy as np
import torch
import paddle
from cvt_torch import cls_cvt
from cvt_torch import default 

from config import get_config
from cvt import build_cvt as build_model

paddle.set_device('cpu')


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
    mapping = [('stage2.cls_token', 'stage2.cls_token')]

    # torch 'layers' to  paddle 'stages'
    depths = [1, 2, 10]
    num_stages = len(depths)
    for stage_idx in range(num_stages):
        pp_s_prefix = f'stage{stage_idx}'
        th_s_prefix = f'stage{stage_idx}'
        layer_mapping = [
            (f'{th_s_prefix}.patch_embed.proj', f'{pp_s_prefix}.patch_embed.proj'),
            (f'{th_s_prefix}.patch_embed.norm', f'{pp_s_prefix}.patch_embed.norm'),
        ] 
        mapping.extend(layer_mapping)

        for block_idx in range(depths[stage_idx]):
            th_b_prefix = f'{th_s_prefix}.blocks.{block_idx}'
            pp_b_prefix = f'{pp_s_prefix}.blocks.{block_idx}'
            layer_mapping = [
                (f'{th_b_prefix}.norm1', f'{pp_b_prefix}.norm1'),
                (f'{th_b_prefix}.attn.conv_proj_q.conv', f'{pp_b_prefix}.attn.conv_proj_q.0'),
                (f'{th_b_prefix}.attn.conv_proj_q.bn', f'{pp_b_prefix}.attn.conv_proj_q.1'),
                (f'{th_b_prefix}.attn.conv_proj_k.conv', f'{pp_b_prefix}.attn.conv_proj_k.0'),
                (f'{th_b_prefix}.attn.conv_proj_k.bn', f'{pp_b_prefix}.attn.conv_proj_k.1'),
                (f'{th_b_prefix}.attn.conv_proj_v.conv', f'{pp_b_prefix}.attn.conv_proj_v.0'),
                (f'{th_b_prefix}.attn.conv_proj_v.bn', f'{pp_b_prefix}.attn.conv_proj_v.1'),
                (f'{th_b_prefix}.attn.proj_q', f'{pp_b_prefix}.attn.proj_q'),
                (f'{th_b_prefix}.attn.proj_k', f'{pp_b_prefix}.attn.proj_k'),
                (f'{th_b_prefix}.attn.proj_v', f'{pp_b_prefix}.attn.proj_v'),
                (f'{th_b_prefix}.attn.proj', f'{pp_b_prefix}.attn.proj'),
                (f'{th_b_prefix}.norm2', f'{pp_b_prefix}.norm2'),
                (f'{th_b_prefix}.mlp.fc1', f'{pp_b_prefix}.mlp.fc1'),
                (f'{th_b_prefix}.mlp.fc2', f'{pp_b_prefix}.mlp.fc2'),
            ]
            mapping.extend(layer_mapping)

    mapping.extend([
        ('norm', 'norm'),
        ('head', 'head')])
    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, no_transpose=False):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
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

    for name, param in paddle_model.named_buffers():
        pd_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()


    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            if th_name.endswith('relative_position_bias_table'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
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
    paddle_config = get_config('./configs/cvt-13-224x224.yaml')
    paddle_model = build_model(paddle_config)
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    torch_config = default.get_config('./CvT/experiments/imagenet/cvt/cvt-13-224x224.yaml')
    torch_model = cls_cvt.get_cls_model(torch_config)
    state_dict = torch.load('./cvt_torch/CvT-13-224x224-IN-1k.pth', map_location=lambda storage, loc: storage)
    torch_model.load_state_dict(state_dict)
    torch_model = torch_model.to(device)
    torch_model.eval()
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, 224, 224).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torch_model(x_torch)
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:20])
    print(out_paddle[0, 0:20])
    assert np.allclose(out_torch, out_paddle, atol = 1e-4)

    # save weights for paddle model
    model_path = os.path.join('./cvt_13_new.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()





#config = default.get_config('./CvT/experiments/imagenet/cvt/cvt-13-224x224.yaml')
#model = cls_cvt.get_cls_model(config)
##print(model)
#
#print('==========')
#state_dict = torch.load('./cvt_torch/CvT-13-224x224-IN-1k.pth', map_location=lambda storage, loc: storage)
#model.load_state_dict(state_dict)
#for key, val in state_dict.items():
#    print(key, val.shape)
#print('==========')
#
#
#
#
#paddle_config = get_config('./configs/cvt-13-224x224.yaml')
#paddle_model = build_model(paddle_config)
##print(paddle_model)
#for key, val in paddle_model.state_dict().items():
#    print(key, val.shape)
#print('==========')
#
#
#
#model.eval()
#paddle_model.eval()
#
#
##r = np.random.randn(2, 3, 224, 224)
##
##tt = torch.Tensor(r)
##
##torch_out = model(tt)
##print(torch_out.shape, torch_out)
#
#
#
#
