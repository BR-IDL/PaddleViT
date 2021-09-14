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

import os
import argparse
import numpy as np
import paddle
import torch
from config import get_config
from pvtv2_det import build_pvtv2_det
from model_utils import DropPath
from utils import NestedTensor
from coco import make_coco_transforms
from PIL import Image
import paddle.vision.transforms as T
import paddle.nn.functional as F

config = get_config('./configs/pvtv2_b0.yaml')

def main():
    # read image
    #image = Image.open('./demo.jpg').convert('RGB')
    #print(image.size) # 380, 640
    #transforms = make_coco_transforms('val')
    #image, _ = transforms(image, None)
    #print(type(image))
    #print(image.shape) # 3, 791, 1332
    
    with open('demo.npy', 'rb') as infile:
        image = np.load(infile)

    image = paddle.to_tensor(image)
    print(image.shape)

    ## pad with divisor 32 tmp:
    #img_w, img_h = image.shape[2], image.shape[1]
    #size_divisor = 32
    #pad_w = int(np.ceil(img_w / size_divisor)) * size_divisor
    #pad_h = int(np.ceil(img_h / size_divisor)) * size_divisor
    #padding = [pad_h - img_h, pad_w - img_w]
#   # image = image.unsqueeze(0)
    #padded_image = F.pad(image, [0, padding[1], 0, padding[0]], value=0.)

    #print('padded image: ', padded_image)

    #target = target.copy()
    #target['size'] = np.array(padded_image.size[::-1], dtype='float32')
    #if 'masks' in target:
    #    target['masks'] = T.pad(target['masks'], (0, padding[0], 0, padding[1]))



    #image = paddle.unsqueeze(image, 0)
    #image = padded_image
    mask = paddle.zeros_like(image)

    pp_in = NestedTensor(image, mask)

    gt = {'imgs_shape': [paddle.to_tensor([640, 380])],
          'scale_factor_wh': [paddle.to_tensor([1332/640, 791/380])]}

    paddle.set_device('gpu')
    paddle_model  = build_pvtv2_det(config)
    paddle_model.eval()

    model_state_dict = paddle.load('./pvtv2_b0_maskrcnn.pdparams')
    paddle_model.set_dict(model_state_dict)

    out_paddle = paddle_model(pp_in, gt)
    #print('paddle_out = ', out_paddle)

if __name__ == "__main__":
    main()
