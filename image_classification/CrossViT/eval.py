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
import cv2
import numpy as np
import torch

from image_classification.CrossViT.models.crossvit import *
from image_classification.CrossViT.paddle_crossvit.paddle_crossvit import *

cls_root='/media/misa/软件/CrossViT复现日记/ImageNetDataset/ILSVRC2012_img_val/'
txt_path='/media/misa/软件/CrossViT复现日记/ImageNetDataset/val_list.txt'

def main():
    paddle.set_device('gpu')
    paddle_model = pd_crossvit_base_224()
    state_dict=paddle.load('port_weights/pd_crossvit_base_224.pdparams')
    paddle_model.load_dict(state_dict)
    paddle_model.eval()
    # print_model_named_params(paddle_model)
    # print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')

    with open(txt_path,'r') as f:
        im_labels=f.readlines()

    imgs=[]
    labels=[]
    for item in im_labels:
        im_name,lbl=item.strip().split(' ')
        imgs.append(im_name)
        labels.append(int(lbl))
    print(len(imgs),len(labels))

    correct=0
    cnt=0

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    for im_name,label in zip(imgs,labels):
        # print(im_name,label)
        x = cv2.imread(cls_root+im_name)
        x = cv2.resize(x, (224, 224))/255.

        x = x-IMAGENET_DEFAULT_MEAN
        x = x /IMAGENET_DEFAULT_STD
        x = x.transpose((2,0,1))

        x = np.expand_dims(x, axis=0).astype('float32')
        x_paddle = paddle.to_tensor(x)
        out_paddle = paddle_model(x_paddle)
        out_paddle = out_paddle.cpu().numpy()
        cnt+=1
        print(cnt)
        if np.argmax(out_paddle)==label:
            correct+=1
    val_acc=(correct/len(labels))
    print(f"ImageNet val acc: {val_acc}")
    print('done!')



if __name__ == "__main__":
    main()
