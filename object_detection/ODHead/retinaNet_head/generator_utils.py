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
"""Anchor generator for retinaNet head"""

import os
import sys

from scipy.cluster.vq import kmeans
import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn as nn

from box_ops import bbox_overlaps

class BaseAnchorCluster(object):
    """
        Base Anchor Cluster
        Args:
            n (int): number of clusters
            cache_path (str): cache directory path
            cache (bool): whether using cache
            verbose (bool): whether print results
    """
    def __init__(self, n, cache_path, cache, verbose=True):
        super(BaseAnchorCluster, self).__init__()
        self.n = n
        self.cache_path = cache_path
        self.cache = cache
        self.verbose = verbose

    def print_result(self, centers):
        '''
            print_result will print the important information.
            But this function should be implement in sub class.
        '''
        raise NotImplementedError('%s.print_result is not available' %
                                  self.__class__.__name__)

    def get_whs(self):
        raise NotImplementedError('%s.calc_anchors is not available' %
                                  self.__class__.__name__)

    def calc_anchors(self):
        raise NotImplementedError('%s.calc_anchors is not available' %
                                  self.__class__.__name__)

    def __call__(self):
        self.get_whs()
        centers = self.calc_anchors()
        if self.verbose:
            self.print_result(centers)
        return centers


class YOLOv5AnchorCluster(BaseAnchorCluster):
    """
        YOLOv5 Anchor Cluster
        Reference:
            https://github.com/ultralytics/yolov5/blob/master/utils/general.py
        Args:
            n (int): number of clusters
            dataset (DataSet): DataSet instance, VOC or COCO
            size (list): [w, h]
            cache_path (str): cache directory path
            cache (bool): whether using cache
            iters (int): iters of kmeans algorithm
            gen_iters (int): iters of genetic algorithm
            threshold (float): anchor scale threshold
            verbose (bool): whether print results
    """
    def __init__(self,
                 n,
                 dataset,
                 size,
                 cache_path,
                 cache,
                 iters=300,
                 gen_iters=1000,
                 thresh=0.25,
                 verbose=True):
        super(YOLOv5AnchorCluster, self).__init__(
            n, cache_path, cache, verbose=verbose)
        self.dataset = dataset
        self.size = size
        self.iters = iters
        self.gen_iters = gen_iters
        self.thresh = thresh

    def print_result(self, centers):
        whs = self.whs
        centers = centers[np.argsort(centers.prod(1))]
        x, best = self.metric(whs, centers)
        bpr, aat = (
            best > self.thresh).mean(), (x > self.thresh).mean() * self.n
        print(
            'thresh=%.2f: %.4f best possible recall, %.2f anchors past thr' %
            (self.thresh, bpr, aat))
        print(
            f'n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thresh=%.3f-mean: '
            % (self.n, self.size, x.mean(), best.mean(),
               x[x > self.thresh].mean()))
        print('%d anchor cluster result: [w, h]' % self.n)
        for w, h in centers:
            print('[%d, %d]' % (round(w), round(h)))

    def metric(self, whs, centers):
        r = whs[:, None] / centers[None]
        x = np.minimum(r, 1. / r).min(2)
        return x, x.max(1)

    def fitness(self, whs, centers):
        _, best = self.metric(whs, centers)
        return (best * (best > self.thresh)).mean()

    def get_whs(self):
        whs_cache_path = os.path.join(self.cache_path, 'whs.npy')
        shapes_cache_path = os.path.join(self.cache_path, 'shapes.npy')
        if self.cache and os.path.exists(whs_cache_path) and os.path.exists(
                shapes_cache_path):
            self.whs = np.load(whs_cache_path)
            self.shapes = np.load(shapes_cache_path)
            return self.whs, self.shapes
        whs = np.zeros((0, 2))
        shapes = np.zeros((0, 2))
        self.dataset.parse_dataset()
        roidbs = self.dataset.roidbs
        for rec in tqdm(roidbs):
            h, w = rec['h'], rec['w']
            bbox = rec['gt_bbox']
            wh = bbox[:, 2:4] - bbox[:, 0:2] + 1
            wh = wh / np.array([[w, h]])
            shape = np.ones_like(wh) * np.array([[w, h]])
            whs = np.vstack((whs, wh))
            shapes = np.vstack((shapes, shape))

        if self.cache:
            os.makedirs(self.cache_path, exist_ok=True)
            np.save(whs_cache_path, whs)
            np.save(shapes_cache_path, shapes)

        self.whs = whs
        self.shapes = shapes
        return self.whs, self.shapes

    def calc_anchors(self):
        self.whs = self.whs * self.shapes / self.shapes.max(
            1, keepdims=True) * np.array([self.size])
        wh0 = self.whs
        i = (wh0 < 3.0).any(1).sum()
        if i:
            print('Extremely small objects found. %d of %d'
                           'labels are < 3 pixels in width or height' %
                           (i, len(wh0)))

        wh = wh0[(wh0 >= 2.0).any(1)]
        print('Running kmeans for %g anchors on %g points...' %
                    (self.n, len(wh)))
        s = wh.std(0)
        centers, _ = kmeans(wh / s, self.n, iter=self.iters)
        centers *= s

        f, sh, mp, s = self.fitness(wh, centers), centers.shape, 0.9, 0.1
        pbar = tqdm(
            range(self.gen_iters),
            desc='Evolving anchors with Genetic Algorithm')
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():
                v = ((np.random.random(sh) < mp) * np.random.random() *
                     np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
            new_centers = (centers.copy() * v).clip(min=2.0)
            new_f = self.fitness(wh, new_centers)
            if new_f > f:
                f, centers = new_f, new_centers.copy()
                pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f

        return centers


def label_box(anchors,
              gt_boxes,
              positive_overlap,
              negative_overlap,
              allow_low_quality,
              ignore_thresh,
              is_crowd=None):
    iou = bbox_overlaps(gt_boxes, anchors)
    n_gt = gt_boxes.shape[0]
    if n_gt == 0 or is_crowd is None:
        n_gt_crowd = 0
    else:
        n_gt_crowd = paddle.nonzero(is_crowd).shape[0]
    if iou.shape[0] == 0 or n_gt_crowd == n_gt:
        # No truth, assign everything to background
        default_matches = paddle.full((iou.shape[1], ), 0, dtype='int64')
        default_match_labels = paddle.full((iou.shape[1], ), 0, dtype='int32')
        return default_matches, default_match_labels
    # if ignore_thresh > 0, remove anchor if it is closed to
    # one of the crowded ground-truth
    if n_gt_crowd > 0:
        N_a = anchors.shape[0]
        ones = paddle.ones([N_a])
        mask = is_crowd * ones

        if ignore_thresh > 0:
            crowd_iou = iou * mask
            valid = (paddle.sum((crowd_iou > ignore_thresh).cast('int32'),
                                axis=0) > 0).cast('float32')
            iou = iou * (1 - valid) - valid

        # ignore the iou between anchor and crowded ground-truth
        iou = iou * (1 - mask) - mask

    matched_vals, matches = paddle.topk(iou, k=1, axis=0)
    match_labels = paddle.full(matches.shape, -1, dtype='int32')
    # set ignored anchor with iou = -1
    neg_cond = paddle.logical_and(matched_vals > -1,
                                  matched_vals < negative_overlap)
    match_labels = paddle.where(neg_cond,
                                paddle.zeros_like(match_labels), match_labels)
    match_labels = paddle.where(matched_vals >= positive_overlap,
                                paddle.ones_like(match_labels), match_labels)
    if allow_low_quality:
        highest_quality_foreach_gt = iou.max(axis=1, keepdim=True)
        pred_inds_with_highest_quality = paddle.logical_and(
            iou > 0, iou == highest_quality_foreach_gt).cast('int32').sum(
                0, keepdim=True)
        match_labels = paddle.where(pred_inds_with_highest_quality > 0,
                                    paddle.ones_like(match_labels),
                                    match_labels)

    matches = matches.flatten()
    match_labels = match_labels.flatten()

    return matches, match_labels
