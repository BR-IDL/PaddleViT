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
# limitations under the License.:

import unittest
import os
import paddle
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from coco import build as build_coco
from coco import make_coco_transforms
from coco import CocoDetection
from box_ops import box_cxcywh_to_xyxy
from pycocotools.coco import COCO
from utils import collate_fn
from utils import NestedTensor
from detr import build_detr


class DetrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #cls.data_path = '/dataset/coco/'
        #cls.coco_dataset_train = build_coco('train', cls.data_path, False)

        #cls.im_mean = np.array([0.485, 0.456, 0.406])
        #cls.im_std = np.array([0.229, 0.224, 0.225]) 
        #cls.coco_dataset_val_det = build_coco('val', CocoTest.data_path, False)
        #cls.coco_dataset_train_det_mask = build_coco('train', CocoTest.data_path, True)
        #cls.coco_dataset_val_det_mask = build_coco('val', CocoTest.data_path, True)
        #cls.cat_train = cls.coco_dataset_train_det.coco.dataset['categories']
        #cls.cat_val = cls.coco_dataset_val_det.coco.dataset['categories']
        #cls.fnt = ImageFont.truetype("./FreeMono.ttf", 20)
        #cls.out = './tmp_out'
        #if not os.path.exists(cls.out):
        #    os.mkdir(cls.out)
        #cls.colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

        with open('./t.npy', 'rb') as infile:
            t = np.load(infile, allow_pickle=True)
            m = np.load(infile, allow_pickle=True)
            tar = np.load(infile, allow_pickle=True)
        #tt = torch.Tensor(t)
        #mm = torch.Tensor(m)
        #th_in = th_utils.NestedTensor(tt, mm)

        # targets
        targets = []
        for ta in tar:
            target = dict()
            for key, val in ta.items():
                target[key] = paddle.to_tensor(val)
            targets.append(target)
        targets = tuple(targets)

        
        ttt = paddle.to_tensor(t)
        mmm = paddle.to_tensor(m)
        pp_in = NestedTensor(ttt, mmm)
        
        #print(th_in, th_in.tensors.shape)
        #print(pp_in, pp_in.tensors.shape)
        #print(targets)

        #cls.th_in = th_in
        cls.pp_in = pp_in
        cls.pp_gt = targets


    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip for debug')
    def test_build_detr_cpu(self):
        paddle.set_device('cpu')
        model, criterion, postprocessors = build_detr()
        model.eval()

        #sampler_train = paddle.io.BatchSampler(CocoTest.coco_dataset_train,
        #                                       batch_size=4,
        #                                       shuffle=False)
        #dataloader_train = paddle.io.DataLoader(CocoTest.coco_dataset_train,
        #                                        batch_sampler=sampler_train,
        #                                        collate_fn=collate_fn)

        #for idx, batch_data in enumerate(dataloader_train):
        #    samples, targets = batch_data[0], batch_data[1]

        #    print('=================================')
        #    print(samples)
        #    print(targets)
        #    paddle.save({'tensors':samples.tensors, 'mask': samples.mask}, path='./batch_samples_01.pdtensor', protocol=2)
        #    paddle.save(targets, path='./batch_targets_01.pdtensor', protocol=2)
        #    print('=================================')
        #    break

        ###    print('----- mask shape = ')
        ###    print(samples.mask.shape)
        ###    print('----- samples shape=')
        ###    print(samples.tensors.shape)
        ###    out = model(samples)
        ###    print(out)

        #samples = paddle.load(path='./batch_samples_01.pdtensor')
        #samples = NestedTensor(samples['tensors'], samples['mask'])
        #print(samples.tensors.shape)
        #targets = paddle.load(path='./batch_targets_01.pdtensor')

        out = model(DetrTest.pp_in)
        #print(out)  
        #print(out.keys()) # pred_logits[2, 100, 92], pred_boxes: [2, 100, 4], aux_outputs

        # test criterion
        losses = criterion(out, DetrTest.pp_gt)
        #print(losses)
        
        return

        ## test postprocess
        target_sizes = [
                        [768, 1027],
                        [768, 1027],
                        [768, 1027],
                        [768, 1027],
                        ] 
        target_sizes = paddle.to_tensor(target_sizes)
        res = postprocessors(outputs, target_size=target_size)
        print(res)




    #@unittest.skip('skip for debug')
    #def test_build_coco_train_det_gpu(self):
    #    paddle.set_device('gpu')
    #    self._test_build_coco_det(CocoTest.coco_dataset_train_det, 'train')

    #@unittest.skip('skip for debug')
    #def test_build_coco_train_det_mask_cpu(self):
    #    paddle.set_device('cpu')
    #    self._test_build_coco_det_mask(CocoTest.coco_dataset_train_det_mask, 'train')

    #@unittest.skip('skip for debug')
    #def test_build_coco_train_det_mask_gpu(self):
    #    paddle.set_device('gpu')
    #    self._test_build_coco_det_mask(CocoTest.coco_dataset_train_det_mask, 'train')

    #@unittest.skip('skip for debug')
    #def test_build_coco_val_det_cpu(self):
    #    paddle.set_device('cpu')
    #    self._test_build_coco_det(CocoTest.coco_dataset_val_det, 'val')

    #@unittest.skip('skip for debug')
    #def test_build_coco_val_det_gpu(self):
    #    paddle.set_device('gpu')
    #    self._test_build_coco_det(CocoTest.coco_dataset_val_det, 'val')

    #@unittest.skip('skip for debug')
    #def test_build_coco_val_det_mask_cpu(self):
    #    paddle.set_device('cpu')
    #    self._test_build_coco_det_mask(CocoTest.coco_dataset_val_det_mask, 'val')

    #@unittest.skip('skip for debug')
    #def test_build_coco_val_det_mask_gpu(self):
    #    paddle.set_device('gpu')
    #    self._test_build_coco_det_mask(CocoTest.coco_dataset_val_det_mask, 'val')

    #def _test_build_coco_det_mask(self, coco_dataset, mode):
    #    for idx, (image, target) in enumerate(coco_dataset):
    #        if 'masks' in target:
    #            masks = target['masks'].cpu().numpy()  # [N, H, W]
    #            if np.any(masks):
    #                print('saving masks into png')
    #                for i in range(masks.shape[0]):
    #                    mask = masks[i, :, :] * 255.0
    #                    mask = mask.astype('uint8')
    #                    im = Image.fromarray(mask)
    #                    im.save(os.path.join(CocoTest.out, f'mask_{mode}_{idx}_{i}_{paddle.get_device()}.png'))

    #                # save image
    #                image = image.transpose([1, 2, 0])  # [C, H, W]
    #                image = image.cpu().numpy()
    #                image = (image * CocoTest.im_std + CocoTest.im_mean) * 255.0
    #                image = image.astype('uint8')
    #                im = Image.fromarray(image)
    #                im.save(os.path.join(CocoTest.out, f'img_mask_{mode}_{idx}_from_{paddle.get_device()}.png'))
    #                break
    #            else:
    #                print('no masks in curren image, continue')
    #                continue
    #        else:
    #            print('no masks in curren image, continue')
    #            continue

    #def _test_build_coco_det(self, coco_dataset, mode):
    #    def get_cat_name(id, cat):
    #        for item in cat: 
    #            if item['id'] == id:
    #                return item['name']
    #        return ""
    #    # used to recover image
    #    for idx, (image, target) in enumerate(coco_dataset):
    #        # recover and save image to file, for manual check
    #        image = image.transpose([1, 2, 0])  # [C, H, W]
    #        image = image.cpu().numpy()
    #        image = (image * CocoTest.im_std + CocoTest.im_mean) * 255.0
    #        image = image.astype('uint8')
    #        im = Image.fromarray(image)
    #        # get bbox labels
    #        labels = target['labels'].cpu().numpy()
    #        # draw bbox on image
    #        h, w = image.shape[0], image.shape[1]
    #        boxes = target['boxes']
    #        boxes = boxes * paddle.to_tensor([w, h, w, h])
    #        boxes = box_cxcywh_to_xyxy(boxes)
    #        boxes = boxes.cpu().numpy()  # [N, 4]
    #        im1 = ImageDraw.Draw(im)
    #        for i in range(boxes.shape[0]):
    #            box = boxes[i].astype('int32')
    #            box = [(box[0], box[1]), (box[2], box[3])]
    #            im1.rectangle(box, outline=CocoTest.colors[i % len(CocoTest.colors)], width=5)
    #            im1.text(box[0], get_cat_name(labels[i], CocoTest.cat_val), font=CocoTest.fnt, fill='red')
    #        im.save(os.path.join(CocoTest.out, f'img_{mode}_{idx}_from_{paddle.get_device()}.png'))
    #        if idx >= 5:
    #            break

