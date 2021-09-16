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

from image_classification.CrossViT.models.crossvit import *
from image_classification.CrossViT.crossvit import *


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


def perpare_mapping(paddle_model, torch_model):
    mapping = []
    for (name, param), (name2, param2) in zip(paddle_model.named_parameters(), torch_model.named_parameters()):
        layer_mapping = [
            (name2, name)
        ]
        mapping.extend(layer_mapping)
    return mapping


def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, transpose=True):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape)  # paddle shape default type is list
        # assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
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
    mapping = perpare_mapping(paddle_model, torch_model)

    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys():  # nn.Parameters
            _set_value(th_name, pd_name)
        else:  # weight & bias
            th_name_w = f'{th_name}.weight'
            pd_name_w = f'{pd_name}.weight'
            _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def pd_gen_model_test():
    paddle.set_device('cpu')
    # paddle_model = pd_crossvit_tiny_224()
    # paddle_model = pd_crossvit_small_224()
    # paddle_model = pd_crossvit_9_224()
    # paddle_model = pd_crossvit_15_224()
    # paddle_model = pd_crossvit_18_224()
    # paddle_model = pd_crossvit_9_dagger_224()
    # paddle_model = pd_crossvit_15_dagger_224()
    # paddle_model = pd_crossvit_15_dagger_384()
    # paddle_model = pd_crossvit_18_dagger_224()
    paddle_model = pd_crossvit_18_dagger_384()
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)


if __name__ == "__main__":
    # main()
    pd_gen_model_test()
