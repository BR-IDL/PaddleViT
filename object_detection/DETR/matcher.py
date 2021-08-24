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

"""
Hugraian matching algorithm for predictions and targets
"""

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Layer):
    def __init__(self, cost_class=1., cost_bbox=1., cost_giou=1.):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict contains 'pred_logits' and 'pred_boxes'
                pred_logits: [batch_size, num_queires, num_classes]
                pred_boxes: [batch_size, num_queires, 4]
            targets: list(tuple) of targets, len(targets) = batch_size, each target is a dict contains at least 'labels' and 'bboxes'
                labels: [num_target_boxes], containing the class labels
                boxes: [num_target_boxes, 4], containing the gt bboxes
        """
        with paddle.no_grad():
            batch_size, num_queries = outputs['pred_logits'].shape[:2]
            # outputs: [batch_size , num_queries , num_classes]
            # pred_boxes: [batch_size , num_queries , 4
            #print('========= orig pred boxes ======')
            #print(outputs['pred_boxes'])

            out_prob = F.softmax(outputs['pred_logits'].flatten(0, 1), -1) # [batch_size*num_queries, num_classes]
            out_bbox = outputs['pred_boxes'].flatten(0, 1) #[batch_size*num_queries, 4]

            tgt_idx = paddle.concat([v['labels'] for v in targets])
            tgt_bbox = paddle.concat([v['boxes'] for v in targets])
            
            ## SAME
            #print('out_bbox', out_bbox, out_bbox.shape)
            #print('tgt_bbox,', tgt_bbox, tgt_bbox.shape)

            cost_class = -paddle.index_select(out_prob, tgt_idx, axis=1)
            #print('cost_class = ', cost_class)

            #cost_bbox = paddle.cdist(out_bbox, tgt_bbox, p=1) # TODO: impl paddle cdist for tensors
            # conver back to numpy for temp use
            out_bbox = out_bbox.cpu().numpy()
            tgt_bbox = tgt_bbox.cpu().numpy()
            cost_bbox = distance.cdist(out_bbox, tgt_bbox, 'minkowski', p=1).astype('float32')
            cost_bbox = paddle.to_tensor(cost_bbox)

            out_bbox = paddle.to_tensor(out_bbox)
            tgt_bbox = paddle.to_tensor(tgt_bbox)

            # SAME
            #print('cost_bbox, ', cost_bbox.shape)
            #print('cost_bbox =', cost_bbox)
            
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            #SAME
            #print('cost_giou', cost_giou, cost_giou.shape)

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.reshape([batch_size, num_queries, -1])
            sizes = [len(v['boxes']) for v in targets]

            # When sizes = [0, n] (no boxes)
            # pytorch C.split(sizes, -1)[0][0] returns: tensor([], size=(100, 0))
            # but paddle C.split(sizes, -1)[0][0] raises error
            # original code in pytorch:
            #idxs = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # We fix for paddle:
            idxs = []
            for i, c in enumerate(C.split(sizes, -1)):
                if c.shape[-1] == 0:
                    idx = linear_sum_assignment(paddle.empty((c.shape[1], c.shape[2])))
                else:
                    idx = linear_sum_assignment(c[i])
                idxs.append(idx)


            #SAME
            #print('idxs=', idxs)

            return [(paddle.to_tensor(i, dtype='int64'), paddle.to_tensor(j, dtype='int64')) for i,j in idxs]


def build_matcher():
    return HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

