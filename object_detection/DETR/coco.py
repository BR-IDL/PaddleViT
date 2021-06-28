import os
import numpy as np
from PIL import Image
import paddle
import transforms as T
from pycocotools import mask as coco_mask
from utils import collate_fn


class CocoDetection(paddle.io.Dataset):
    def __init__(self, img_folder, anno_file, transforms, return_masks):
        super(CocoDetection, self).__init__()
        from pycocotools.coco import COCO
        self.coco = COCO(anno_file)
        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = self._remove_images_without_annotations(ids)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMasks(return_masks)
        self.root = img_folder

    def _remove_images_without_annotations(self, ids):
        new_ids = []
        rm_cnt = 0
        for id in ids:
            annos = self._load_target(id)
            boxes = []
            for anno in annos:
                if 'bbox' in anno:
                    boxes.append(anno['bbox'])
            if len(boxes) == 0:
                rm_cnt += 1
                continue
            new_ids.append(id)
        print(f'loading coco data, {rm_cnt} imgs without annos are removed')
        return new_ids


    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]['file_name']
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        target = {'image_id': image_id, 'annotations': target}
        
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # paddle any only support bool type
        mask = paddle.to_tensor(mask, dtype='bool') # w x h x 1
        mask = mask.any(axis=2).squeeze(-1) # w x h
        # paddle stack does not support bool type
        mask = mask.astype('int32')
        masks.append(mask)
    if masks:
        masks = paddle.stack(masks, axis=0)
    else:
        mask = paddle.zeros((0, height, width), dtype='int32')
    return masks


class ConvertCocoPolysToMasks():
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target['image_id']
        image_id = paddle.to_tensor([image_id])

        anno = target['annotations']
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj['bbox'] for obj in anno]
        boxes = paddle.to_tensor(boxes, dtype='float32')
        boxes = boxes.reshape([-1, 4])
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clip_(min=0, max=w)
        boxes[:, 1::2].clip_(min=0, max=h)

        classes = [obj['category_id'] for obj in anno]
        classes = paddle.to_tensor(classes, dtype='int64')

        if self.return_masks:
            segmentations = [obj['segmentation'] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)  # [N, H, W] int32 tensor

        keypoints = None
        if anno and 'keypoints' in anno[0]:
            keypoints = [obj['keypoints'] for obj in anno]
            keypoints = paddle.to_tensor(keypoints, dtype='float32')
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape_((num_keypoints, -1, 3))

        #TODO: should be replaced with paddle buildin logical ops in the future
        boxes_tmp = boxes.cpu().numpy()
        keep = (boxes_tmp[:, 3] > boxes_tmp[:, 1]) & (boxes_tmp[:, 2] > boxes_tmp[:, 0])
        keep_idx = np.where(keep)[0]
        keep = paddle.to_tensor(keep_idx)

        boxes = boxes.index_select(keep, axis=0)
        classes = classes.index_select(keep, axis=0)
        if self.return_masks:
            masks = masks.index_select(keep, axis=0)
        if keypoints is not None:
            keypoints = keypoints.index_select(keep, axis=0)

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        if self.return_masks:
            target['masks'] = masks
        if keypoints is not None:
            target['keypoints'] = keypoints
        target['image_id'] = image_id

        area = paddle.to_tensor([obj['area'] for obj in anno])
        iscrowd = paddle.to_tensor([obj['iscrowd'] if 'iscrowd' in obj else 0 for obj in anno])
        target['area'] = area
        target['iscrowd'] = iscrowd.index_select(keep, axis=0)

        target['orig_size'] = paddle.to_tensor([int(h), int(w)])
        target['size'] = paddle.to_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
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
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'Unknown {image_set}')


def build_coco(image_set, coco_path, masks=False):
    root = coco_path
    assert os.path.exists(root), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        'train': (os.path.join(root, 'train2017'), os.path.join(root, 'annotations', f'{mode}_train2017.json')),
        'val': (os.path.join(root, 'val2017'), os.path.join(root, 'annotations', f'{mode}_val2017.json')),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, anno_file, transforms=make_coco_transforms(image_set), return_masks=masks)
    return dataset


def get_loader(dataset, batch_size, mode='train', multi_gpu=False):
    if multi_gpu is True:
        sampler = paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=True if mode is 'train' else False,
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
                                          shuffle=True if mode is 'train' else False,
                                          collate_fn=collate_fn)

    return dataloader
