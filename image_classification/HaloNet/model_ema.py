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

""" Implement the Exponential Model Averaging
This is paddle hack from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
"""

import copy
import paddle


class ModelEma:
    """Model Ema
    A moving average is kept of model weights and buffers.
    Note that for multiple gpu, ema must be defined after mode init,
    but before DataParallel.

    Args:
        model: nn.Layer, original modela with learnable params
        decay: float, decay rate for each update, default: 0.999
    """
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @paddle.no_grad()
    def _update(self, model, update_fn):
        # update ema model parameters by model parameters
        for (_, ema_param), (_, model_param) in zip(
            self.module.named_parameters(), model.named_parameters()):
            ema_param.set_value(copy.deepcopy(update_fn(ema_param, model_param)))

        # update ema model buffers by model buffers
        for (_, ema_buf), (_, model_buf) in zip(
            self.module.named_buffers(), model.named_buffers()):
            ema_buf.set_value(copy.deepcopy(update_fn(ema_buf, model_buf).astype(model_buf.dtype)))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e  + (1 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def to(self, device):
        self.module.to(device)

    def state_dict(self):
        return self.module.state_dict()
