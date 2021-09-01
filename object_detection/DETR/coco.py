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
Dataset(COCO2017) related classes and methods for DETR training and validation
"""

import os
import numpy as np
from PIL import Image
import paddle
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import transforms as T
from utils import collate_fn


class CocoDetection(paddle.io.Dataset):
    """ COCO Detection dataset

    This class gets images and annotations for paddle training and validation.
    Transform(preprocessing) can be applied in __getitem__ method.

    Attributes:
        img_folder: str, file path of coco images, e.g.{COCO_PATH}/train2017
        anno_file: str, path where annotation json file is stored
        transforms: transforms applied on data, see make_coco_transform for details
        return_masks: bool, if true, return coco masks, default: False (now only support False)
    """

    def __init__(self, img_folder, anno_file, transforms, return_masks):
        super(CocoDetection, self).__init__()
        self.coco = COCO(anno_file)
        # coco all image ids
        ids = list(sorted(self.coco.imgs.keys()))
        # remove ids where anno has no bboxes
        self.ids = self._remove_images_without_annotations(ids)
        self._transforms = transforms
        # Prepare has __call__ method takes image and target as inputs
        # and applies label filtering and output image and label as paddle tensors
        self.prepare = ConvertCocoPolysToMasks(return_masks)
        self.root = img_folder

    def _remove_images_without_annotations(self, ids):
        new_ids = []
        rm_cnt = 0
        for idx in ids:
            annos = self._load_target(idx)
            boxes = []
            for anno in annos:
                if 'bbox' in anno:
                    boxes.append(anno['bbox'])
            if len(boxes) == 0:
                rm_cnt += 1
                continue
            new_ids.append(idx)
        print(f'loading coco data, {rm_cnt} imgs without annos are removed')
        return new_ids

    def _load_image(self, idx):
        """ Return PIL Image (RGB) according to COCO image id"""
        path = self.coco.loadImgs(idx)[0]['file_name']
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def _load_target(self, idx):
        """ Return image annos according to COCO image id"""
        return self.coco.loadAnns(self.coco.getAnnIds(idx))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """idx is for training image id, not COCO image id"""
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        target = {'image_id': image_id, 'annotations': target}

        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    """ Convert coco anno from polygons to image masks"""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]

        mask = mask.any(axis=2).squeeze(-1)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        mask = np.zeros((0, height, width), dtype='int32')
    return masks


class ConvertCocoPolysToMasks():
    """ Prepare coco annotations to paddle tensors"""
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target['image_id']
        # Cuda may raise error, use cpu tensor instead
        #image_id = paddle.to_tensor([image_id]).cpu()

        anno = target['annotations']
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]

        ## Temp Fix: do it in numpy to skip paddle cuda error
        boxes = np.array(boxes, dtype='float32')
        boxes = boxes.reshape([-1, 4])
        boxes[:, 2:] += boxes[:, :2] # (n, (x1, y1, x2, y2))

        boxes[:, 0::2].clip(0, w) # clip bbox inside image
        boxes[:, 1::2].clip(0, h) # clip bbox inside image

        classes = [obj['category_id'] for obj in anno]
        classes = np.array(classes, dtype='float32')

        if self.return_masks:
            segmentations = [obj['segmentation'] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)  # [N, H, W] int32 tensor

        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = np.array(keypoints, dtype='float32')
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((num_keypoints, -1, 3))

        #TODO: should be replaced with paddle buildin logical ops in the future
        boxes_tmp = boxes
        keep = (boxes_tmp[:, 3] > boxes_tmp[:, 1]) & (boxes_tmp[:, 2] > boxes_tmp[:, 0])
        keep_idx = np.where(keep)[0].astype('int32')

        boxes = boxes[keep]
        classes = classes[keep]

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        if self.return_masks:
            target['masks'] = masks
        if keypoints is not None:
            target['keypoints'] = keypoints
        target['image_id'] = image_id

        area = np.array([obj['area'] for obj in anno], dtype='float32')
        iscrowd = np.array([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in anno], dtype='float32')
        target['area'] = area
        target['iscrowd'] = iscrowd[keep]

        target['orig_size'] = np.array([int(h), int(w)], dtype='float32')
        target['size'] = np.array([int(h), int(w)], dtype='float32')

        return image, target


def make_coco_transforms(image_set):
    """ return transforms(class defined in ./transforms.py) for coco train and val"""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'Unknown {image_set}')


def build_coco(image_set, coco_path, masks=False):
    """Return CocoDetection dataset according to image_set: ['train', 'val']"""
    assert image_set in ['train', 'val'], f'image_set {image_set} not supported'
    assert os.path.exists(coco_path), f'provided COCO path {coco_path} does not exist'
    mode = 'instances'
    paths = {
        'train': (os.path.join(coco_path, 'train2017'),
                  os.path.join(coco_path, 'annotations', f'{mode}_train2017.json')),
        'val': (os.path.join(coco_path, 'val2017'),
                os.path.join(coco_path, 'annotations', f'{mode}_val2017.json')),
    }
    img_folder, anno_file = paths[image_set]
    dataset = CocoDetection(img_folder,
                            anno_file,
                            transforms=make_coco_transforms(image_set),
                            return_masks=masks)
    return dataset


def get_dataloader(dataset, batch_size, mode='train', multi_gpu=False):
    """ return dataloader on train/val set for single/multi gpu
    Arguments:
        dataset: paddle.io.Dataset, coco dataset
        batch_size: int, num of samples in one batch
        mode: str, ['train', 'val'], dataset to use
        multi_gpu: bool, if True, DistributedBatchSampler is used for DDP
    """
    if multi_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            drop_last=True)
        #TODO: may need to fix this drop_last of multi-gpu dataloading error
        # currently, val may drop several samples, which will lower the performance
        # an idea is to pad the last batch in collate_fn
        dataloader = paddle.io.DataLoader(dataset,
                                          batch_sampler=sampler,
                                          collate_fn=collate_fn)
    else:
        dataloader = paddle.io.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=(mode == 'train'),
                                          collate_fn=collate_fn)
    return dataloader
