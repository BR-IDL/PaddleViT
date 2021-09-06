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


import json
import tempfile
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(dataset, predictions):
    coco_results = {}
    coco_results['bbox'] = make_coco_detection(predictions, dataset)

    results = COCOResult('bbox')

    with tempfile.NamedTemporaryFile() as f:
        path = f.name
        res = evaluate_predictions_on_coco(
            dataset.coco, coco_results['bbox'], path, 'bbox'
        )
        results.update(res)

    return results


def evaluate_predictions_on_coco(coco_gt, results, result_file, iou_type):
    with open(result_file, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(str(result_file)) if results else COCO()

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # compute_thresholds_for_classes(coco_eval)

    return coco_eval


def compute_thresholds_for_classes(coco_eval):
    precision = coco_eval.eval['precision']
    precision = precision[0, :, :, 0, -1]
    scores = coco_eval.eval['scores']
    scores = scores[0, :, :, 0, -1]

    recall = np.linspace(0, 1, num=precision.shape[0])
    recall = recall[:, None]

    f1 = (2 * precision * recall) / (np.maximum(precision + recall, 1e-6))
    max_f1 = f1.max(0)
    max_f1_id = f1.argmax(0)
    scores = scores[max_f1_id, range(len(max_f1_id))]

    print('Maximum f1 for classes:')
    print(list(max_f1))
    print('Score thresholds for classes')
    print(list(scores))


def make_coco_detection(predictions, dataset):
    coco_results = []

    for id, pred in predictions.items():

        if len(pred) == 0:
            continue

        labels = pred[:, 0]
        scores = pred[:, 1]
        boxes = pred[:, 2:]
             
        boxes[:, 2:4] -= boxes[:, 0:2] # convert to x0y0wh

        boxes = boxes.numpy()
        scores = scores.numpy()
        labels = labels.numpy()

        mask = (labels != -1)
        boxes = boxes[mask].tolist()
        scores = scores[mask].tolist()
        labels = labels[mask].tolist()
        
        labels = [dataset.ids2cats[i] for i in labels]

        coco_results.extend(
            [
                {
                    'image_id': id,
                    'category_id': labels[k],
                    'bbox': box,
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    return coco_results


class COCOResult:
    METRICS = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'box_proposal': [
            'AR@100',
            'ARs@100',
            'ARm@100',
            'ARl@100',
            'AR@1000',
            'ARs@1000',
            'ARm@1000',
            'ARl@1000',
        ],
        'keypoints': ['AP', 'AP50', 'AP75', 'APm', 'APl'],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResult.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResult.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        return repr(self.results)