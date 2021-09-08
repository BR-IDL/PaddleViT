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


import math

import paddle
import paddle.nn as nn
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core

class AnchorGenerator(nn.Layer):
    """
    Compute anchors in the standard ways described in
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".

    Attributes:
        anchor_size (list[list[float]] | list[float]):
            If ``anchor_size`` is list[list[float]], ``anchor_size[i]`` is the list of anchor sizes
            (i.e. sqrt of anchor area) to use for the i-th feature map.
            If ``anchor_size`` is list[float], ``anchor_size`` is used for all feature maps.
            Anchor anchor_size are given in absolute lengths in units of
            the input image; they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
            (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
        strides (list[int]): stride of each input feature.
        offset (float): Relative offset between the center of the first anchor and the top-left
            corner of the image. Value has to be in [0, 1).
            Recommend to use 0.5, which means half stride.
    """

    def __init__(self, 
                 anchor_sizes = [[32], [64], [128], [256], [512]],
                 aspect_ratios = [0.5, 1.0, 2.0],
                 strides = [4, 8, 16, 32, 64],
                 offset = 0.5):
        super(AnchorGenerator, self).__init__()

        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.offset = offset
        self.base_anchors = self._compute_anchors()

        assert 0. <= self.offset <= 1.0

    def generate_anchors(self, 
                        sizes = [32, 64, 128, 256, 512], 
                        aspect_ratios = [0.5, 1.0, 2.0]):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).
        Args:
            sizes (list[float] | tuple[float]):
            aspect_ratios (list[float] | tuple[float]]):
        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in xyxy format.
        """
        anchors = []
        
        for size in sizes:
            area = size ** 2.0
            for ratio in aspect_ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        
        return paddle.to_tensor(anchors, dtype='float32')
    
    def _broadcast_params(self, params, num_features):
        if not isinstance(params[0], (list, tuple)):
            return [params] * num_features
        if len(params) == 1:
            return params * num_features
        return params
        
    def _compute_anchors(self):
        sizes = self._broadcast_params(self.anchor_sizes, len(self.strides))
        aspect_ratios = self._broadcast_params(self.aspect_ratios, len(self.strides))

        base_anchors = [self.generate_anchors(s, a) for s, a in zip(sizes, aspect_ratios)]

        [self.register_buffer(t.name, t, persistable=False) for t in base_anchors]

        return base_anchors

    def _grid_anchors(self, grid_sizes):
        anchors = []

        for grid_size, stride, base_anchor in zip(grid_sizes, self.strides, self.base_anchors):
            grid_h, grid_w = grid_size

            grid_x = paddle.arange(
                self.offset * stride, grid_w * stride, step = stride, dtype='float32'
            )
            grid_y = paddle.arange(
                self.offset * stride, grid_h * stride, step = stride, dtype='float32'
            )

            grid_y, grid_x = paddle.meshgrid(grid_y, grid_x)
            grid_x = grid_x.reshape([-1])
            grid_y = grid_y.reshape([-1])

            grid_coord = paddle.stack([grid_x, grid_y, grid_x, grid_y], axis=1)

            anchors.append((grid_coord.unsqueeze(1) + base_anchor.unsqueeze(0)).reshape([-1, 4]))

        return anchors
    
    def forward(self, feats):
        grid_sizes = [feat.shape[-2:] for feat in feats]
        anchor_over_all_feat_maps = self._grid_anchors(grid_sizes)

        return anchor_over_all_feat_maps
    
    @property
    def num_anchors(self):
        return [len(num_a) for num_a in self.base_anchors][0]

# feats = []
# h, w = 800., 800
# for i in range(4):
#     feats.append(paddle.rand([4, 256, h / (2 ** (i + 2)), w / (2 ** (i + 2))]))

# anchorgenerator = AnchorGenerator()
# res = anchorgenerator(feats)
# print(anchorgenerator.num_anchors)
# print(res)
def generate_proposals(scores,
                       bbox_deltas,
                       im_shape,
                       anchors,
                       variances,
                       pre_nms_top_n=6000,
                       post_nms_top_n=1000,
                       nms_thresh=0.5,
                       min_size=0.1,
                       eta=1.0,
                       pixel_offset=False,
                       return_rois_num=False,
                       name=None):
    """
    **Generate proposal Faster-RCNN**
    This operation proposes RoIs according to each box with their
    probability to be a foreground object and 
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.
    For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates. 
    3. Clip boxes to image
    4. Remove predicted boxes with small area. 
    5. Apply NMS to get final proposals as output.

    Args:
        scores (tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas (tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_shape (tensor): A 2-D Tensor with shape [N, 2] represents H, W, the
            origin image size or input size. The data type can be float32 or 
            float64.
        anchors (tensor): A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4] or [H * W * A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances (tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in (xcenter, ycenter, w, h) format. 
            The data type must be float32.
        pre_nms_top_n (float): Number of total bboxes to be kept per image before NMS. 
            The data type must be float32. `6000` by default.
        post_nms_top_n (float): Number of total bboxes to be kept per image after NMS. The data type must be float32. 
            `1000` by default.
        nms_thresh (float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size (float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num (bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's 
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model. 
            'False' by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 
    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. 
            The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. 
            The data type is the same as ``scores``.
    """
    assert in_dygraph_mode()
    assert return_rois_num, "return_rois_num should be True in dygraph mode."
    attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta,
                'pixel_offset', pixel_offset)
    # print('scores:', scores)
    #print('bbox_deltas:', bbox_deltas)
    #print('im_shape:', im_shape)
    #print('anchors:', anchors)
    #print('variances:', variances)
    rpn_rois, rpn_roi_probs, rpn_rois_num = core.ops.generate_proposals_v2(
        scores, bbox_deltas, im_shape, anchors, variances, *attrs)

    return rpn_rois, rpn_roi_probs, rpn_rois_num


class ProposalGenerator(object):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Attributes:
        pre_nms_top_n (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.Default 6000
        post_nms_top_n (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.Default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        eta (float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
             `adaptive_threshold = adaptive_threshold * eta` in each iteration.
             default 1.
        topk_after_collect (bool): whether to adopt topk after batch 
             collection. If topk_after_collect is true, box filter will not be 
             used after NMS at each image in proposal generation. default false
    """

    def __init__(self,
                 pre_nms_top_n = 6000,
                 post_nms_top_n = 1000,
                 nms_thresh = .5,
                 min_size = .1,
                 eta = 1.,
                 topk_after_collect = False):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta
        self.topk_after_collect = topk_after_collect

    def __call__(self, scores, bbox_deltas, anchors, imgs_shape):
        top_n = self.pre_nms_top_n if self.topk_after_collect else self.post_nms_top_n
        variances = paddle.ones_like(anchors)
        rpn_rois, rpn_rois_prob, rpn_rois_num = generate_proposals(
            scores,
            bbox_deltas,
            imgs_shape,
            anchors,
            variances,
            pre_nms_top_n=self.pre_nms_top_n,
            post_nms_top_n=top_n,
            nms_thresh=self.nms_thresh,
            min_size=self.min_size,
            eta=self.eta,
            return_rois_num=True
        )

        return rpn_rois, rpn_rois_prob, rpn_rois_num, self.post_nms_top_n  


def roi_align(input,
              rois,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              aligned=True):
    """
    Region of interest align (also known as RoI align) is to perform
    bilinear interpolation on inputs of nonuniform sizes to obtain 
    fixed-size feature maps (e.g. 7*7).

    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        rois (Tensor): ROIs (Regions of Interest) to pool over.It should be
            a 2-D Tensor or 2-D LoDTensor of shape (num_rois, 4), the lod level is 1. 
            The data type is float32 or float64. Given as [[x1, y1, x2, y2], ...],
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        output_size (list[int, int] | tuple[int, int]): The pooled output size(h, w), data type is int32.
        spatial_scale (list[float32], optional): Multiplicative spatial scale factor to translate ROI coords 
            from their input scale to the scale used when pooling. Default: 1.0
        sampling_ratio(int32, optional): number of sampling points in the interpolation grid. 
            If <=0, then grid points are adaptive to roi_width and pooled_w, likewise for height. Default: -1
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:
        Output: The output of ROIAlignOp is a 4-D tensor with shape (num_rois, channels, pooled_h, pooled_w).
            The data type is float32 or float64.
    """

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio, "aligned", aligned)

        return align_out


def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             pixel_offset=False,
                             rois_num=None):
    """
    
    **This op only takes LoDTensor as input.** In Feature Pyramid Networks 
    (FPN) models, it is needed to distribute all proposals into different FPN 
    level, with respect to scale of the proposals, the referring scale and the 
    referring level. Besides, to restore the order of proposals, we return an 
    array which indicates the original index of rois in current proposals. 

    Args:
        fpn_rois(tensor): 2-D Tensor with shape [N, 4] and data type is 
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come 
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default. 

    Returns:
        Tuple:
        multi_rois(list[tensor]) : A list of 2-D LoDTensor with shape [M, 4] 
        and data type of float32 and float64. The length is 
        max_level-min_level+1. The proposals in each FPN level.
        restore_ind(tensor): A 2-D Tensor with shape [N, 1], N is 
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.
        rois_num_per_level(list(tensor)): A list of 1-D Tensor and each Tensor is 
        the RoIs' number in each image on the corresponding level. The shape 
        is [B] and data type of int32. B is the number of images.

    """
    num_lvl = max_level - min_level + 1

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale, 'pixel_offset',
                 pixel_offset)
        multi_rois, restore_ind, rois_num_per_level = core.ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)

        return multi_rois, restore_ind, rois_num_per_level


class RoIAlign(object):
    '''
    Region of interest feature map pooler that supports pooling from 
    one or more feature maps.
    '''
    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        canonical_box_size=224,
        canonical_level=4,
        min_level=0,
        max_level=3,
        aligned=True
    ):
        '''
        Attributes:
            output_size (int): output size of the pooled region.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.
                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
            start_level (int): The start level of FPN layer to extract RoI feature, default 0.
            end_level (int): The end level of FPN layer to extract RoI feature, default 3.
            aligned (bool): Whether to add offset to rois' coord in roi_align. default True.
        '''
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.scales = scales
        self.sampling_ratio = sampling_ratio
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self.min_level = min_level
        self.max_level = max_level
        self.aligned = aligned
    
    def __call__(self, feats, rois, rois_num):
        '''
        Args:
            feats (list[tensor]): features from fpn.
            rois (list[tensor]): proposals from rpn.
            rois_num (list[int]): the number of each img's proposals.
        
        Returns:
            roi_features (tensor): A tensor of shape (M, C, output_size, output_size)
            where M is the total number of boxes aggregated over all N batch images
            and C is the number of channels in `x`.
        '''
        if isinstance(rois_num, list):
            rois_num = paddle.to_tensor(rois_num).astype("int32")
        rois = paddle.concat(rois)

        if len(feats) == 1:
            roi_features = roi_align(
                feats[self.min_level],
                rois,
                self.output_size,
                self.scales[0],
                self.sampling_ratio,
                rois_num=rois_num,
                aligned=self.aligned
            )

        else:
            rois_per_level, original_ind, rois_num_per_level = distribute_fpn_proposals(
                rois,
                self.min_level + 2,
                self.max_level + 2,
                self.canonical_level,
                self.canonical_box_size,
                rois_num=rois_num
            )

            roi_features_per_level = []

            for l in range(self.min_level, self.max_level + 1):
                roi_feats = roi_align(
                    feats[l],
                    rois_per_level[l],
                    self.output_size,
                    self.scales[l],
                    self.sampling_ratio,
                    rois_num=rois_num_per_level[l],
                    aligned = self.aligned
                )

                roi_features_per_level.append(roi_feats)
            
            roi_features = paddle.gather(
                paddle.concat(roi_features_per_level),
                original_ind
            )
        
        return roi_features

