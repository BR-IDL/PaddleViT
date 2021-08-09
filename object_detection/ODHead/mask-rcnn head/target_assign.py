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
from .box_utils import boxes_iou, bbox2delta

def anchor_target_matcher(match_quality_matrix, 
                          positive_thresh,
                          negative_thresh,
                          allow_low_quality_matches,
                          low_thresh = -float("inf")):
    '''
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    Args:
        match_quality_matrix (tensor): an MxN tensor, containing the pairwise quality 
            between M ground-truth elements and N predicted elements.
        positive_thresh (float): the positive class threshold of iou between anchors and gt.
        negative_thresh (float): the negative class threshold of iou between anchors and gt.
        allow_low_quality_matches (bool): if True, produce additional matches
            for predictions with maximum match quality lower than high_threshold.
    
    Returns:
        matches (tensor): a vector of length M, where matches[i] is a matched
            ground-truth index in [0, M).
        match_labels (tensor): a vector of length M, where pred_labels[i] indicates
            whether a prediction is a true or false positive or ignored.
        
    '''
    # matches is 1 x M, the index of anchors matching gt
    matched_vals, matches = paddle.topk(match_quality_matrix, k = 1, axis = 0)
    match_labels = paddle.full(matches.shape, -1, dtype = "int32")
    neg_idx = paddle.logical_and(matched_vals > low_thresh,
                                 matched_vals < negative_thresh)
    match_labels = paddle.where(matched_vals >= positive_thresh,
                                paddle.ones_like(match_labels), 
                                match_labels)
    match_labels = paddle.where(neg_idx,
                                paddle.zeros_like(match_labels), 
                                match_labels)

    # highest_quality_foreach_gt is N x 1
    # For each gt, find the prediction with which it has highest quality
    if allow_low_quality_matches:
        highest_quality_foreach_gt = match_quality_matrix.max(axis=1, keepdim=True)
        pred_inds_with_highest_quality = paddle.logical_and(
        match_quality_matrix >= 0, match_quality_matrix == highest_quality_foreach_gt).cast('int32').sum(
            0, keepdim=True)
        match_labels = paddle.where(pred_inds_with_highest_quality > 0,
                                    paddle.ones_like(match_labels),
                                    match_labels)

    matches = matches.flatten()
    match_labels = match_labels.flatten()

    return matches, match_labels


# reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/sampling.py
def subsample_labels(labels,
                     num_samples,
                     positive_fraction,
                     bg_label=0):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (tensor): shape (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = paddle.nonzero(paddle.logical_and(labels != -1, labels != bg_label))
    negative = paddle.nonzero(labels == bg_label)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    if num_pos == 0 and num_neg == 0:
        pos_idx = paddle.zeros([0], dtype='int32')
        neg_idx = paddle.zeros([0], dtype='int32')
        return pos_idx, neg_idx

    # randomly select positive and negative examples
    negative = negative.cast('int32').flatten()
    neg_perm = paddle.randperm(negative.numel(), dtype='int32')[:int(num_neg[0])]
    neg_idx = paddle.gather(negative, neg_perm)

    if num_pos == 0:
        pos_idx = paddle.zeros([0], dtype='int32')
        return pos_idx, neg_idx

    positive = positive.cast('int32').flatten()
    pos_perm = paddle.randperm(positive.numel(), dtype='int32')[:int(num_pos)]
    pos_idx = paddle.gather(positive, pos_perm)

    return pos_idx, neg_idx
    

def anchor_target_assign(anchors,
                         gt_boxes,
                         positive_thresh,
                         negative_thresh,
                         batch_size_per_image,
                         positive_fraction,
                         allow_low_quality_matches=False,
                         is_crowd=None,
                         weights=[1., 1., 1., 1.]):
    '''
    Args:
        anchors (tensor): shape [-1, 4] the sum of muti-level anchors.
        gt_boxes (list): gt_boxes[i] is the i-th img's gt_boxes.
        positive_thresh (float): the positive class threshold of iou between anchors and gt.
        negative_thresh (float): the negative class threshold of iou between anchors and gt.
        batch_size_per_image (int): number of anchors per image to sample for training.
        positive_fraction (float): fraction of foreground anchors to sample for training.
        allow_low_quality_matches (bool): if True, produce additional matches
            for predictions with maximum match quality lower than high_threshold.
        is_crowd (list | None): is_crowd[i] is is_crowd label of the i-th img's gt_boxes.
        weights (list): more detail please see bbox2delta.

    Returns:
        tgt_labels (list[tensor]): tgt_labels[i].shape is [Ni], the label(positive or negative) of anchors.
        tgt_bboxes (list[tensor]): tgt_bboxes[i].shape is [Ni, 4], the matched gt_boxes.
        tgt_deltas (list[tensor]): tgt_deltas[i].shape is [Ni, 4], the deltas between anchors and gt_boxes.
    '''
    tgt_labels = []
    tgt_bboxes = []
    tgt_deltas = []

    low_thresh = -float("inf")
    for i in range(len(gt_boxes)):
        gt_bbox = gt_boxes[i]
        n_gt = gt_bbox.shape[0]
        
        if n_gt == 0 or is_crowd is None:
            n_is_crowd = 0 
        else:
            is_crowd_i = is_crowd[i]
            n_is_crowd = paddle.nonzero(is_crowd_i).shape[0]

        match_quality_matrix, _ = boxes_iou(gt_bbox, anchors)
        assert match_quality_matrix.dim() == 2
        
        # ignore the iou between anchor and crowded ground-truth
        if n_is_crowd > 0:
            n_a = anchors.shape[0]
            ones = paddle.ones([n_a])
            mask = is_crowd_i * ones
            match_quality_matrix = match_quality_matrix * (1 - mask) - mask
            low_thresh = -1
        # match_quality_matrix is N (gt) x M (predicted)
        # assert (match_quality_matrix >= 0).all()
        if match_quality_matrix.shape[0] == 0 or n_gt == n_is_crowd:
            matches = paddle.full((match_quality_matrix.shape[1], ), 0, dtype='int64')
            match_labels = paddle.full((match_quality_matrix.shape[1], ), 0, dtype='int32')
        else:
            matches, match_labels = anchor_target_matcher(match_quality_matrix,
                                                          positive_thresh,
                                                          negative_thresh,
                                                          allow_low_quality_matches,
                                                          low_thresh)
        
        pos_idx, neg_idx = subsample_labels(match_labels, 
                                            batch_size_per_image, 
                                            positive_fraction)

        # Fill with the ignore label (-1), then set positive and negative labels
        labels = paddle.full(match_labels.shape, -1, dtype='int32')
        if neg_idx.shape[0] > 0:
            labels = paddle.scatter(labels, neg_idx, paddle.zeros_like(neg_idx))
        if pos_idx.shape[0] > 0:
            labels = paddle.scatter(labels, pos_idx, paddle.ones_like(pos_idx))
        
        if n_gt == 0:
            matched_gt_boxes = paddle.zeros([0, 4])
            tgt_delta = paddle.zeros([0, 4])
        else:
            matched_gt_boxes = paddle.gather(gt_bbox, matches)
            tgt_delta = bbox2delta(anchors, matched_gt_boxes, weights)
            matched_gt_boxes.stop_gradient = True
            tgt_delta.stop_gradient = True

        labels.stop_gradient = True
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)

    return tgt_labels, tgt_bboxes, tgt_deltas


def roi_target_assign(proposals,
                      gt_boxes,
                      gt_classes,
                      num_classes,
                      positive_thresh,
                      negative_thresh,
                      batch_size_per_image,
                      positive_fraction,
                      allow_low_quality_matches=False):
    '''
    It performs box matching between "roi" and "target",and assigns training labels
    to the proposals. 

    Args:
        proposals (list[tensor]): the batch RoIs from rpn_head.
        gt_boxes (list[tensor]): gt_boxes[i] is the i'th img's gt_boxes.
        gt_classes (list[tensor]): gt_classes[i] is the i'th img's gt_classes.
        num_classes (int): the number of class.
    
    Returns:
        proposals_info (dict): a dict contains the information of proposals. 
    '''

    proposals_info = {}
    num_fg_samples = []
    proposals_samples = []
    num_proposals = []
    gt_boxes_samples = []
    gt_cls_samples = []

    for proposals_single_img, bbox_single_img, label_single_img in zip(proposals, gt_boxes, gt_classes):
        match_quality_matrix, _ = boxes_iou(bbox_single_img, proposals_single_img)
        matched_idxs, matched_labels = anchor_target_matcher(match_quality_matrix, 
                                                             positive_thresh,
                                                             negative_thresh,
                                                             allow_low_quality_matches)

        if label_single_img.numel() > 0:
            label_single_img = label_single_img.squeeze()
            label_single_img = paddle.gather(label_single_img, matched_idxs)
            label_single_img = paddle.where(matched_labels == 0,
                                            paddle.full_like(label_single_img, num_classes),
                                            label_single_img)

            label_single_img = paddle.where(matched_labels == -1,
                                            paddle.full_like(label_single_img, -1),
                                            label_single_img)
        else:
            label_single_img = paddle.zeros_like(matched_idxs) + num_classes
            sample_gt_box = paddle.zeros_like(bbox_single_img)

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(label_single_img,
                                                            batch_size_per_image,
                                                            positive_fraction,
                                                            num_classes)

        sampled_idxs = paddle.concat([sampled_fg_idxs, sampled_bg_idxs])
        sample_proposal = paddle.gather(proposals_single_img, sampled_idxs)
        sample_gt_cls = paddle.gather(label_single_img, sampled_idxs)

        if label_single_img.numel() > 0:
            sample_box_idx = paddle.gather(matched_idxs, sampled_idxs)
            sample_gt_box = paddle.gather(bbox_single_img, sample_box_idx)

        num_fg_samples.append(sampled_fg_idxs.shape[0])      
        proposals_samples.append(sample_proposal)
        num_proposals.append(sampled_idxs.shape[0])
        gt_boxes_samples.append(sample_gt_box)
        gt_cls_samples.append(sample_gt_cls)
    
    proposals_info["num_fg"] = num_fg_samples
    proposals_info["proposals"] = proposals_samples
    proposals_info["num_proposals"] = num_proposals
    proposals_info["gt_boxes"] = gt_boxes_samples
    proposals_info["gt_classes"] = gt_cls_samples

    return proposals_info
