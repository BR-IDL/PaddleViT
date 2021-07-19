'''
    This module use to convert a pytorch shuffle_transformer model to paddle model.
'''
import os
import argparse
import paddle
import torch
import numpy as np
import shuffle_paddle
import shuffle_transformer
from config import *


config = get_config()
parser = argparse.ArgumentParser('')
parser.add_argument('-cfg',
                    type=str,
                    default='./config/shuffle_vit_tiny_patch4_window7_224.yaml')
#parser.add_argument('-dataset', type=str, default="imagenet2012")
parser.add_argument('-dataset', type=str, default="cifar10")
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-image_size', type=int, default=224)
parser.add_argument('-data_path', type=str, default='/dataset/imagenet/')
parser.add_argument('-eval', action="store_true")
parser.add_argument('-pretrained', type=str, default=None)
args = parser.parse_args()

config = get_config()
config = update_config(config, args)

def build_paddle_model(configs):
    '''
    Describe:
        This function used to build a paddle model.
    Args:
        config---------> a parser object.
    '''
    model = shuffle_paddle.ShuffleTransformer(img_size=configs.DATA.IMAGE_SIZE,
                                              in_chans=configs.MODEL.TRANS.IN_CHANNELS,
                                              num_classes=configs.MODEL.NUM_CLASSES,
                                              embed_dim=configs.MODEL.TRANS.EMBED_DIM,
                                              layers=configs.MODEL.TRANS.STAGE_DEPTHS,
                                              num_heads=configs.MODEL.TRANS.NUM_HEADS,
                                              window_size=configs.MODEL.TRANS.WINDOW_SIZE,
                                              mlp_ratio=configs.MODEL.TRANS.MLP_RATIO,
                                              qkv_bias=configs.MODEL.TRANS.QKV_BIAS,
                                              qk_scale=configs.MODEL.TRANS.QK_SCALE,
                                              drop_rate=configs.MODEL.DROPOUT,
                                              drop_path_rate=configs.MODEL.DROP_PATH
                                              )
    return model

def build_pytorch_model(configs):
    '''
    Describe:
        This function used to build a pytorch model.
    Args:
        config---------> a parser object.
    '''
    model = shuffle_transformer.ShuffleTransformer(img_size=configs.DATA.IMAGE_SIZE,
                                                   in_chans=configs.MODEL.TRANS.IN_CHANNELS,
                                                   num_classes=configs.MODEL.NUM_CLASSES,
                                                   embed_dim=configs.MODEL.TRANS.EMBED_DIM,
                                                   layers=configs.MODEL.TRANS.STAGE_DEPTHS,
                                                   num_heads=configs.MODEL.TRANS.NUM_HEADS,
                                                   window_size=configs.MODEL.TRANS.WINDOW_SIZE,
                                                   mlp_ratio=configs.MODEL.TRANS.MLP_RATIO,
                                                   qkv_bias=configs.MODEL.TRANS.QKV_BIAS,
                                                   qk_scale=configs.MODEL.TRANS.QK_SCALE,
                                                   drop_rate=configs.MODEL.DROPOUT,
                                                   drop_path_rate=configs.MODEL.DROP_PATH)
    return model

def print_model_named_params(model):
    '''
    Describe:
        This function used to print model's parameter.
    Args:
        model-----------> a pytorch or paddle model.
    '''
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_buffer_params(model):
    '''
    Describe:
        This function used to print model's buffer.
    Args:
        model-----------> a pytorch or paddle model.
    '''
    for name, buffer in model.named_buffers():
        print(name, buffer.shape)


def mapping_generagte():
    '''
    Describe:
        This fucntion used to generate a map, which is a refelence between paddle and pytorch.
    '''
    mapping = [
        ('to_token.conv1.0','to_token.conv1.0'),
        ('to_token.conv1.1','to_token.conv1.1'),
        ('to_token.conv2.0','to_token.conv2.0'),
        ('to_token.conv2.1','to_token.conv2.1'),
        ('to_token.conv3','to_token.conv3'),
    ]
    depths = config.MODEL.TRANS.STAGE_DEPTHS
    num_stages = len(depths)
    for stage_index in range(num_stages):
        pp_s_prefix = f'stage{stage_index + 1}.layers'
        th_s_prefix = f'stage{stage_index + 1}.layers'
        if stage_index > 0:
            mapping.extend([
                (f'stage{stage_index + 1}.patch_partition.norm',
                 f'stage{stage_index + 1}.patch_partition.norm'),
                (f'stage{stage_index + 1}.patch_partition.reduction',
                 f'stage{stage_index + 1}.patch_partition.reduction')
            ])
        for block_index_0 in range(depths[stage_index] // 2):
            for block_index_1 in range(2):
                th_b_prefix = f'{th_s_prefix}.{block_index_0}.{block_index_1}'
                pp_b_prefix = f'{pp_s_prefix}.{block_index_0}.{block_index_1}'
                layer_mapping = [
                    (f'{th_b_prefix}.norm1', f'{pp_b_prefix}.norm1'),
                    (f'{th_b_prefix}.attn.relative_position_bias_table',
                     f'{pp_b_prefix}.attn.relative_position_bias_table'),
                    (f'{th_b_prefix}.attn.to_qkv', f'{pp_b_prefix}.attn.to_qkv'),
                    (f'{th_b_prefix}.attn.proj', f'{pp_b_prefix}.attn.proj'),
                    (f'{th_b_prefix}.local', f'{pp_b_prefix}.local'),
                    (f'{th_b_prefix}.norm2', f'{pp_b_prefix}.norm2'),
                    (f'{th_b_prefix}.mlp.fc1', f'{pp_b_prefix}.mlp.fc1'),
                    (f'{th_b_prefix}.mlp.fc2', f'{pp_b_prefix}.mlp.fc2'),
                    (f'{th_b_prefix}.norm3', f'{pp_b_prefix}.norm3'),
                ]
                mapping.extend(layer_mapping)

    mapping.extend([
        ('head', 'head')
    ])
    return mapping





def convert(torch_model, paddle_model):
    '''
    Describe:
        This function used to convert a torch model to a paddle model.
    Args:
        torch_model --------> a pytorch model, which should be convert.
        paddle_model--------> a paddle model, which should be a target model.
    '''
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
    for name, buffer in torch_model.named_buffers():
        th_params[name] = buffer
    # 2. get name mapping pairs
    mapping = mapping_generagte()

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            if th_name.endswith('relative_position_bias_table'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
                _set_value(th_name, pd_name)
        else: # weight & bias
            if th_name.endswith('to_qkv') or th_name.endswith('reduction'):
                _set_value(f'{th_name}.weight', f'{pd_name}.weight')
            else:
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)
                if th_name.endswith('norm') or th_name.endswith('norm1') \
                or th_name.endswith('norm2') or th_name.endswith('norm3') \
                or th_name == 'to_token.conv1.1' or th_name == 'to_token.conv2.1':
                    th_name_m = f'{th_name}.running_mean'
                    th_name_v = f'{th_name}.running_var'
                    pd_name_m = f'{pd_name}._mean'
                    pd_name_v = f'{pd_name}._variance'
                    _set_value(th_name_m, pd_name_m)
                    _set_value(th_name_v, pd_name_v)

    return paddle_model





if __name__ == '__main__':
    print("Enter the main")
    paddle.set_device('cpu')
    paddlemodel = build_paddle_model(config)
    paddlemodel.eval()
    print_model_named_params(paddlemodel)


    device = torch.device('cpu')
    torchmodel  = build_pytorch_model(config)
    torchmodel  = torchmodel.to(device)
    torchmodel.eval()
    modelpath = './shuffle_vit_tiny_patch4_window7_224_local7midV2_drop0.1_82.41/ckpt_epoch_298.pth'
    checkpoint = torch.load(modelpath, map_location='cpu')
    torchmodel.load_state_dict(checkpoint['model'], strict=False)
    print_model_named_params(torchmodel)
    print_model_buffer_params(torchmodel)
    paddlemodel = convert(torchmodel, paddlemodel)


    # check correctness
    x = np.random.randn(1, 3, 224, 224).astype('float32')
    #x = np.ones((1, 3, 224, 224)).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    out_torch = torchmodel(x_torch)
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    out_paddle = paddlemodel(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:20])
    print(out_paddle[0, 0:20])
    assert np.allclose(out_torch, out_paddle, atol = 1e-4)
    model_path = os.path.join('./shuffle_vit_tiny_patch4_window7_224.pdparams')
    paddle.save(paddlemodel.state_dict(), model_path)
