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

import cv2
import numpy as np
import paddle
from config import get_config
from config import update_config
from crossvit import build_crossvit


def print_model_named_params(model):
    """
    model params print
    """
    print('----------------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)
    print('----------------------------------')


def print_model_named_buffers(model):
    """
    buffer params print
    """
    print('----------------------------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
    print('----------------------------------')


def main():
    """
    build model from config,
    image data pre-process,at here we don't sub image-net  mean and divided std,but it doesn't effect the final result
    zerbra.jpg predict id will be 340,if there nothing wrong.
    """
    parser = argparse.ArgumentParser('CrossViT')
    parser.add_argument('-cfg', type=str, default="configs/crossvit_base_224.yaml")
    args = parser.parse_args()
    config = get_config()
    config = update_config(config, args)

    paddle.set_device('cpu')

    paddle_model = build_crossvit(config)
    state_dict = paddle.load('port_weights/pd_crossvit_base_224.pdparams')
    paddle_model.load_dict(state_dict)
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')

    image_x = cv2.imread('zerbra.jpeg')
    resize_x = cv2.resize(image_x, (224, 224)) / 255.
    resize_x = resize_x.transpose((2, 0, 1))
    resize_x = np.expand_dims(resize_x, axis=0).astype('float32')
    print(resize_x.shape)
    x_paddle = paddle.to_tensor(resize_x)
    print(x_paddle.shape)
    out_paddle = paddle_model(x_paddle)
    out_paddle = out_paddle.cpu().numpy()
    print('========================================================')
    print(np.argmax(out_paddle))
    print('done!')


main()
