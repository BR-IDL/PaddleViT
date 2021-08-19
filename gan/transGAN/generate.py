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

"""Generate images using trained models"""
import argparse
import os
from PIL import Image
import paddle
from models.ViT_custom import Generator
from config import get_config
from config import update_config

def main():
    """ generate sample images using pretrained model
    The following args are required:
        -cfg: str, path of yaml model config file
        -pretrained: str, path of the pretrained model (ends with .pdparams)
        -num_out_images: int, the num of output images to be saved in file
        -out_folder: str, output folder path.
    """
    parser = argparse.ArgumentParser('transGAN')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-num_out_images', type=int, default=16)
    parser.add_argument('-out_folder', type=str, default='./out_images_cifar10')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-last_epoch', type=int, default=None)
    args = parser.parse_args()

    # get default config
    config = get_config()
    # update config by arguments
    config = update_config(config, args)

    # get model
    print(f'----- Creating model...')
    paddle_model = Generator(config)
    paddle_model.eval()
    # load model weights
    print(f'----- Loading model form {config.MODEL.PRETRAINED}...')
    model_state_dict = paddle.load(config.MODEL.PRETRAINED)
    paddle_model.load_dict(model_state_dict['gen_state_dict'])
    # get random input tensor
    x_paddle = paddle.randn([args.num_out_images, config.MODEL.LATENT_DIM])
    # inference
    print(f'----- Inferencing...')
    out_paddle = paddle_model(x_paddle, 0)
    # post processing to obtain image
    print('----- Postprocessing')
    gen_imgs = (out_paddle * 127.5 + 128).clip(0,255).transpose((0, 2, 3, 1))
    gen_imgs = gen_imgs.astype('uint8').cpu().numpy()
    # save images to file
    os.makedirs(args.out_folder, exist_ok=True)
    print(f'----- Saving images to {args.out_folder}')
    for i, gen_img in enumerate(gen_imgs):
        img = Image.fromarray(gen_img, 'RGB')
        out_path = os.path.join(args.out_folder, str(i) + '.png')
        img.save(out_path)

if __name__ == "__main__":
    main()