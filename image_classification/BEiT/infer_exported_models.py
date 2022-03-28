# Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softw
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# See the License for the specific language governing permissions
# limitations under the License.
"""
BEiT inference using paddle.inference api

An implementation for BEiT model to load static model,
and do prediction using paddleinference api
"""
import numpy as np
import paddle.inference as paddle_infer
from paddle.vision import image_load
from paddle.vision import transforms


def load_img(img_path, sz=224):
    """load image and apply transforms"""
    if isinstance(sz, int):
        sz = (sz, sz)
    data = image_load(img_path).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize(sz, 'bicubic'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        ])
    data = trans(data)
    return data



def main():
    config = paddle_infer.Config(
        './beit_base_patch16_224.pdmodel',
        './beit_base_patch16_224.pdiparams',
    )
    batch_size = 4
    in_channels = 3
    image_size = 224

    predictor = paddle_infer.create_predictor(config)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    batch_data = np.random.randn(batch_size, in_channels, image_size, image_size).astype('float32')

    ## read images from file
    #data_all = []
    #for i in range(1, 5):
    #    image_path = f'./images/{i}.jpg'
    #    data = load_img(image_path, sz=image_size)
    #    # convert back to numpy, copy_from_cpu need numpy
    #    data = data.numpy()
    #    data_all.append(data)
    #batch_data = np.array(data_all)
    #print(batch_data.shape)
    

    input_handle.reshape([batch_size, in_channels, image_size, image_size])
    input_handle.copy_from_cpu(batch_data)
    
    predictor.run()
    
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    print(output_data)
    print('output shape = ', output_data.shape)


if __name__ == "__main__":
    main()
