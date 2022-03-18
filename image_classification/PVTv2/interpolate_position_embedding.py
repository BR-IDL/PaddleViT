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
"""interpolate position tokens
   interpolate when  number of model's position tokens is not equal to loaded model state dict,
   keep the extra tokens (cls_token, dist_token, etc.) unchanged.
"""
import paddle


def interpolate_position_embedding(model, state_dict):
    """interpolate pos embed from model state for new model"""
    if 'pos_embed' in state_dict:
        pos_embed_state = state_dict['pos_embed']
        embed_dim = pos_embed_state.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # size from loaded model weights
        orig_size = int((pos_embed_state.shape[-2] - num_extra_tokens) ** 0.5)
        # size for new model
        new_size = int(num_patches ** 0.5)
        # if same size, nothing changed
        if orig_size != new_size:
            print(f'Interpolate pos embed from {orig_size}x{orig_size} to {new_size}x{new_size}')
            extra_tokens = pos_embed_state[:, :num_extra_tokens]
            # only interpolate position tokens, cls_token or dist_token are not changed
            pos_tokens = pos_embed_state[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size, embed_dim])
            pos_tokens = pos_tokens.transpose([0, 3, 1, 2]) # [n, d, h, w]
            pos_tokens = paddle.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.transpose([0, 2, 3, 1]) # [n, h, w, d]
            pos_tokens = pos_tokens.flatten(1, 2) # [n, h*w, d]
            new_pos_embed = paddle.concat([extra_tokens, pos_tokens], axis=1)
            state_dict['pos_embed'] = new_pos_embed
