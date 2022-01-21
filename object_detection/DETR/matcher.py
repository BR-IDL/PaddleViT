from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from box_ops import box_cxcywh_to_xyxy
from box_ops import generalized_box_iou


def cdist_p1(x, y):
    # x: [batch * num_queries, 4]
    # y: [batch * num_boxes, 4]
    x = x.unsqueeze(1)
    res = x - y
    res = paddle.norm(res, p=1, axis=-1)
    return res


class HugarianMatcher(nn.Layer):
    def __init__(self, cost_class=1., cost_bbox=1., cost_giou=2.):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @paddle.no_grad()
    def forward(self, outputs, targets):    
        """
        Args:
            outputs: dict contains 'pred_logits' and 'pred_boxes'
                pred_logits: [batch_size, num_queires, num_classes]
                pred_boxes: [batch_size, num_queires, 4]
            targets: list(tuple) of targets, len(targets) = batch_size, each target is a dict contain
                labels: [num_target_boxes], containing the class labels
                boxes: [num_target_boxes, 4], containing the gt bboxes
        """
        batch_size, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = F.softmax(outputs['pred_logits'].flatten(0, 1), axis=-1) #[batch * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1) # [batch * num_queries, 4]

        # TODO: check case when tgt is empty, may be unnecessary
        idx_list = []
        for v in targets: # for each sample label in current batch
            if v['labels'].shape[0] !=0:
                idx_list.append(v['labels'])
        if len(idx_list) > 0: # if current batch has label
            tgt_idx = paddle.concat(idx_list) # tgt_idx contains all the labels in batch
            tgt_idx = tgt_idx.astype('int32')
        else:
            tgt_idx = paddle.empty([0], dtype='int32')

        bbox_list = []
        for v in targets:
            if v['boxes'].shape[0] != 0:
                bbox_list.append(v['boxes'])
        if len(bbox_list) > 0:
            tgt_bbox = paddle.concat(bbox_list)
        else:
            tgt_bbox = paddle.empty([0], dtype='float32')
            
        if tgt_idx.is_empty():
            cost_class = 0
            cost_bbox = 0
            cost_giou = 0
        else:
            # approximate NLL loss to 1-prob[target_class], 1 could be ommitted
            #cost_class: [batch*num_queries, batch_num_boxes]
            cost_class = -paddle.index_select(out_prob, tgt_idx, axis=1)
            #cost_bbox: [batch*num_queries, batch_num_boxes]

            # Option1: my impl using paddle apis
            cost_bbox = cdist_p1(out_bbox, tgt_bbox)
            ## Option2: convert back to numpy
            #out_bbox = out_bbox.cpu().numpy()
            #tgt_bbox = tgt_bbox.cpu().numpy()
            #cost_bbox = distance.cdist(out_bbox, tgt_bbox, 'minkowski', p=1).astype('float32')
            #cost_bbox = paddle.to_tensor(cost_bbox)


            out_bbox = paddle.to_tensor(out_bbox)
            tgt_bbox = paddle.to_tensor(tgt_bbox)
            #cost_giou: [batch*num_queries, batch_num_boxes]
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.reshape([batch_size, num_queries, -1])

        sizes = [len(v['boxes']) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if c.shape[-1] == 0:
                idx = linear_sum_assignment(paddle.empty((c.shape[1], c.shape[2])))
            else:
                idx = linear_sum_assignment(c[i])
            indices.append(idx)

        return [(paddle.to_tensor(i, dtype='int64'),
                 paddle.to_tensor(j, dtype='int64')) for i, j in indices]


def build_matcher():
    return HugarianMatcher(cost_class=1., cost_bbox=5., cost_giou=2.)
