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

"""parameters groups for layer-wise lr decay, used in BeiT and MAE"""

import json

# Note: param_groups_lrd is NOT used because paddle Adam optimizer seems has problems which we don't know,
#       instead, we use paddlenlp.ops.optimizer.AdamWDL with lr_settings (see below) right now for temp fix.
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    """layer-wise decay
    set learning rate decay according to layer depth
    Note: 
        1. In Paddle param_groups, dict key 'learning_rate' is in fact the 'lr_mult'
        2. param_names in no_weight_decay_list will have no decay
        3. model.encoder.layers may need to change for models other than MAE_finetune
    """
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.encoder.layers) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for name, param in model.named_parameters():
        if param.stop_gradient is True:
            continue

        # no decay
        if param.ndim == 1 or name.endswith('.bias') or name in no_weight_decay_list:
            g_decay = 'no_decay'
            this_weight_decay = 0.
        else:
            g_decay = 'decay'
            this_weight_decay = weight_decay

        layer_id = get_layer_id_for_vit(name, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "learning_rate": this_scale,
                "weight_decay": this_weight_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "learning_rate": this_scale,
                "weight_decay": this_weight_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)
        
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """assign a parameter with its layer id"""
    if name in ['cls_token', 'mask_token', 'encoder_position_embedding']:
        return 0
    elif name.startswith('patch_embedding'):
        return 0
    elif name.startswith('encoder.layers'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers


def lr_setting(layer_decay, name_dict, num_layers, param):
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    static_name = name_dict[param.name]
    #print('static_name= ', static_name, ', param.name= ', param.name)
    layer_id = get_layer_id_for_vit(static_name, num_layers)
    param.optimize_attr["learning_rate"] *= layer_scales[layer_id]
