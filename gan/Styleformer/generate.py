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
import numpy
import paddle
from PIL import Image
from generator import Generator
from config import *


def main():
    # get default config
    parser = argparse.ArgumentParser('')
    parser.add_argument('-cfg', type=str, default='./configs/styleformer_cifar10.yaml')
    parser.add_argument('-dataset', type=str, default="cifar10")
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-data_path', type=str, default='/dataset/cifar10/')
    parser.add_argument('-eval', action="store_true")
    parser.add_argument('-pretrained', type=str, default=None)
    args = parser.parse_args()
    config = get_config()
    config = update_config(config, args)

    paddle.set_device('cpu')
    paddle_model = Generator(config)
    paddle_model.eval()

    pre=paddle.load(r'./cifar10.pdparams')
    paddle_model.load_dict(pre)

    x = paddle.randn([32, 512])
    x_paddle = paddle.to_tensor(x)
    out_paddle = paddle_model(x_paddle, c=paddle.randint(0, 10, [32]))

    gen_imgs=paddle.multiply(out_paddle,paddle.to_tensor(127.5))
    gen_imgs=paddle.clip(paddle.add(gen_imgs,paddle.to_tensor(127.5)).transpose((0,2,3,1)),
             min=0.0,max=255.0).astype('uint8').cpu().numpy()

    for i in range(len(gen_imgs)):
        im = Image.fromarray(gen_imgs[i], 'RGB')
        im.save("./image/"+str(i)+".png")


if __name__ == "__main__":
    main()
