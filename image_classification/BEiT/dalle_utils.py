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

import paddle

logit_laplace_eps = 0.1

def map_pixels(x):
    if x.dtype != paddle.float32:
        raise ValueError('expected input to have type float32')

    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x):
    if len(x.shape) != 4:
        raise ValueError('expected input to be 4d')
    if x.dtype != paddle.float32:
        raise ValueError('expected input to have type float32')

    return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)
