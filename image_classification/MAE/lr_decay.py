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

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.encoder.layers) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if p.stop_gradient is True:
            continue

        # no decay
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = 'no_decay'
            this_decay = 0.
        else:
            g_decay = 'decay'
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "learning_rate": this_scale, # TODO: check correctness 
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "learning_rate": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """assign a parameter with its layer id"""
    if name in ['cls_token', 'position_embedding']:
        return 0
    elif name.startswith('patch_embedding'):
        return 0
    elif name.startswith('encoder.layers'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers

