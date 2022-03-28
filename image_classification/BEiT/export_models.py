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
BEiT export to inference model in Paddle

An implementation for BEiT model to save static model,
which can be used in paddleinference
"""
import paddle
from config import get_config
from beit import build_beit as build_model


def main():
    """load .pdparams model file, save to static model:
        .pdiparams, .pdiparams.info, .pdmodel
    """
    model_names = [
        "beit_base_patch16_224",
        "beit_large_patch16_224",
    ]
    model_folder = './'
    out_folder = './'

    for model_name in model_names:
        model_path = f'{model_folder}/{model_name}.pdparams'
        out_path = f'{out_folder}/{model_name}'
        config = get_config(f'./configs/{model_name}.yaml')
        model = build_model(config)
        model_state = paddle.load(model_path)
        if 'model' in model_state:
            if 'model_ema' in model_state:
                model_state = model_state['model_ema']
            else:
                model_state = model_state['model']
        model.set_state_dict(model_state)

        model.eval()

        img_size = [config.DATA.IMAGE_CHANNELS, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE]
        build_strategy = paddle.static.BuildStrategy()
        # some optimized settings
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
        build_strategy.reduce_strategy = paddle.static.BuildStrategy.ReduceStrategy.Reduce
        build_strategy.fuse_broadcast_ops = True
        build_strategy.fuse_elewise_add_act_ops = True

        model = paddle.jit.to_static(
            model,
            input_spec=[paddle.static.InputSpec(shape=[None] + img_size, dtype='float32')],
            build_strategy=build_strategy)

        paddle.jit.save(model, out_path)
        print(f'export model {model_name} done')


if __name__ == "__main__":
    main()

