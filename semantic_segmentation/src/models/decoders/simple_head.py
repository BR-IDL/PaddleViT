#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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
Implement Simple Head Class for Top Transformer
"""

import warnings
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Identity(nn.Layer):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    """
    def forward(self, inputs):
        return inputs


class ConvModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.SyncBatchNorm,
                 act=nn.ReLU,
                 bias_attr=False):
        super(ConvModule, self).__init__()
        
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias_attr=bias_attr)
        self.act = act() if act is not None else Identity()
        self.norm = norm(out_channels) if norm is not None else Identity()


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SimpleHead(nn.Layer):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, config):
        super(SimpleHead, self).__init__()
        self._init_inputs(config.MODEL.SIMPLE.IN_CHANNELS,
                          config.MODEL.SIMPLE.IN_INDEX,
                          config.MODEL.SIMPLE.INPUT_TRANSFORM)
        self.channels = config.MODEL.SIMPLE.CHANNELS
        self.is_dw = config.MODEL.SIMPLE.IS_DW
        self.dropout = nn.Dropout2D(config.MODEL.SIMPLE.DROPOUT_RATIO)
        self.conv_seg = nn.Conv2D(self.channels, config.DATA.NUM_CLASSES , kernel_size=1)
        self.index = config.MODEL.SIMPLE.IN_INDEX
        self.align_corners=config.MODEL.SIMPLE.ALIGN_CORNERS
        self.seg_label_shape = config.DATA.CROP_SIZE

        self.linear_fuse = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            groups=self.channels if self.is_dw else 1,
        )

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = paddle.concat(upsampled_inputs, axis=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    def agg_res(self, preds):
        outs = preds[0]
        for pred in preds[1:]:
            pred = resize(pred, size=outs.shape[2:], mode='bilinear', align_corners=self.align_corners)
            outs += pred
        return outs

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        x = resize(
            input=x,
            size=self.seg_label_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        return x
