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

"""
Positional embeddings, contains classes for sine-based and learning-based implementations.
"""

import copy
import math
import paddle
import paddle.nn as nn


class PositionEmbeddingSine(nn.Layer):
    def __init__(self, num_pos_feats=64, temp=10000, norm=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temp = temp
        self.norm = norm
        if scale is not None and norm is False:
            raise ValueError('norm should be true is scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        
        #print('mask -----')
        #for i  in range(mask.shape[0]):
        #    for j in range(mask.shape[1]):
        #        for k in range(mask.shape[2]):
        #            print(int(mask[i, j, k].cpu().numpy()[0]), end=',')
        #        print()
        #    print('-----')

        not_mask = (mask < 0.5).astype('float32')

        y_embed = not_mask.cumsum(1, dtype='float32')
        x_embed = not_mask.cumsum(2, dtype='float32')
        
        #print('-----y_embed')
        #print(y_embed)

        if self.norm:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype='int32') # paddle elementwise_floordiv support int32
#TODO: check bug

        dim_t = self.temp ** (2 * (dim_t // 2) / self.num_pos_feats) # int32 will cast to float32 

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = paddle.stack((pos_x[:,:,:,0::2].sin(), pos_x[:,:,:,1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack((pos_y[:,:,:,0::2].sin(), pos_y[:,:,:,1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])

        #print('----- pos')
        #print(pos)

        return pos
        

class PositionEmbeddingLearned(nn.Layer):
    def __init__(self, num_pos_feats=256):
        super(PositionEmbeddingLearned, self).__init__()
        w_attr1 = self._init_weights()
        w_attr2 = self._init_weights()
        self.row_embed = nn.Embedding(50, num_pos_feats, weight_attr=w_attr1) #TODO: why 50? maximum?
        self.col_embed = nn.Embedding(50, num_pos_feats, weight_attr=w_attr2)

    def _init_weights(self):
        return paddle.ParamAttr(initializer=nn.initializer.Uniform(low=0., high=1.))

    def forward(self, tensor_list):
        x = tensor_list.tensors # [batch, 2048(R50 feat), H, W]
        h, w = x.shape[-2:]
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)

        #print('x_embed, ', x_embed.shape)
        #print('y_embed, ', y_embed.shape)
    
        pos = paddle.concat([
            x_embed.unsqueeze(0).expand((h, x_embed.shape[0], x_embed.shape[1])),
            y_embed.unsqueeze(1).expand((y_embed.shape[0], w, y_embed.shape[1])),
            ], axis=-1)
        #print(pos.shape)
        pos = pos.transpose([2, 0, 1]) # [dim, h, w]
        pos = pos.unsqueeze(0) # [1, dim, h, w]
        pos = pos.expand([x.shape[0]] + pos.shape[1::]) # [batch_size, dim, h, w]

        return pos


def build_position_encoding(hidden_dim=256, mode='sine'):
    N_steps = hidden_dim // 2 
    if mode == 'sine':
        position_embedding = PositionEmbeddingSine(N_steps, norm=True)
    elif mode == 'learned':
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f'{mode} not supported')
    return position_embedding
