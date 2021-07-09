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


"""
Fold operation, which is usually equivalent to 'col2im' operation.
Current paddle version (2.1) is not supported native Fold operation.
This hack is based on for-loop, which may be optimized in the future.
"""

import numpy as np
import paddle


def fold(inputs, output_size, kernel_size, padding, stride):
    """
    Args:
        x: Tensor, input tensor, only support 3D tensor, [Batch, C * kernel_size * kernel_size, L]
        output_size, Tuple/List, contains the height and width of the output tensor, len = 2
        kernel_size: int, kernel size
        padding: int, num of pad around the input
        stride: int, stride for sliding window
    """

    B, D, L = inputs.shape
    H, W = output_size
    C = int(D / (kernel_size * kernel_size))
    out_h = (H + 2*padding -kernel_size) // stride + 1
    out_w = (W + 2*padding -kernel_size) // stride + 1

    inputs = inputs.reshape([B, C, kernel_size, kernel_size, out_h, out_w])
    
    img = paddle.zeros([B, C, H + 2 * padding + stride - 1, W + 2 * padding + stride -1], dtype=inputs.dtype)

    for y in range(kernel_size):
        y_max = y + stride * out_h
        for x in range(kernel_size):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += inputs[:, :, y, x, :, :]

    return img[:, :, padding: H + padding, padding: W + padding]



#def main():
#    paddle.set_device('cpu')
#    arr = [
#        [1, 1, 1, 1, 2, 2, 2, 2],
#        [1, 1, 1, 1, 2, 2, 2, 2],
#        [1, 1, 1, 1, 2, 2, 2, 2],
#        [1, 1, 1, 1, 2, 2, 2, 2],
#        [3, 3, 3, 3, 4, 4, 4, 4],
#        [3, 3, 3, 3, 4, 4, 4, 4],
#        [3, 3, 3, 3, 4, 4, 4, 4],
#        [3, 3, 3, 3, 4, 4, 4, 4],
#    ]
#    arr = np.array(arr)
#    tmp = paddle.to_tensor(arr, dtype='float32')
#    tmp = tmp.reshape([1, 1, 8, 8])
#
#    unfold = paddle.nn.Unfold(3, 1, 1)
#    out = unfold(tmp)
#
#    for i in range(out.shape[-1]):
#        row = out[:, :, i].astype('int8').numpy()
#        print(row)
#    out = fold(out, output_size=(8, 8), kernel_size=3, padding=1, stride=1)
#    print(out)
#
#if __name__ == "__main__":
#    main()
#
#
## k=3, p=2, s=2
##[[[4. , 2. , 4. , 2. , 8. , 4. , 8. , 4. ],
##   2. , 1. , 2. , 1. , 4. , 2. , 4. , 2. ],
##   4. , 2. , 4. , 2. , 8. , 4. , 8. , 4. ],
##   2. , 1. , 2. , 1. , 4. , 2. , 4. , 2. ],
##   12., 6. , 12., 6. , 16., 8. , 16., 8. ],
##   6. , 3. , 6. , 3. , 8. , 4. , 8. , 4. ],
##   12., 6. , 12., 6. , 16., 8. , 16., 8. ],
##   6. , 3. , 6. , 3. , 8. , 4. , 8. , 4. ]]]])
#
#
## k = 3, p=1, s=1
## [[[[4. , 6. , 6. , 6. , 12., 12., 12., 8. ],
##     [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
##     [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
##     [6. , 9. , 9. , 9. , 18., 18., 18., 12.],
##     [18., 27., 27., 27., 36., 36., 36., 24.],
##     [18., 27., 27., 27., 36., 36., 36., 24.],
##     [18., 27., 27., 27., 36., 36., 36., 24.],
##     [12., 18., 18., 18., 24., 24., 24., 16.]]]])
##
#
