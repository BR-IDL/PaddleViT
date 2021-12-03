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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class VIT_MLA_AUXIHead(nn.Layer):
    """VIT_MLA_AUXIHead
 
    VIT_MLA_AUXIHead is the auxiliary segmentation decoder of SETR-MLA
    Reference:                                                                                                                                                
        Sixiao Zheng, et al. *"Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"*
    """

    def __init__(self, in_channels = 256, num_classes = 60, align_corners= False):
        super(VIT_MLA_AUXIHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        if self.in_channels == 1024:
            self.aux_0 = nn.Conv2D(self.in_channels, 256, 
                                   kernel_size=1, bias_attr=False)
            self.aux_1 = nn.Conv2D(256, self.num_classes, 
                                   kernel_size=1, bias_attr=False)
        elif self.in_channels == 256:
            self.aux = nn.Conv2D(self.in_channels, self.num_classes, 
                                 kernel_size=1, bias_attr=False)


    def forward(self, x):
        up16x_resolution = [ 16*item for item in x.shape[2:]]
        if self.in_channels == 1024:
            x = self.aux_0(x)
            aux_pred = self.aux_1(x)
        elif self.in_channels == 256:
            aux_pred = self.aux(x)
        aux_pred_full = F.interpolate(
            aux_pred, up16x_resolution, mode='bilinear', 
            align_corners=self.align_corners)
        return aux_pred_full
