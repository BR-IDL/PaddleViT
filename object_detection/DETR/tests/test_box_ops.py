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
import paddle
import numpy as np
from box_ops import *
from utils import NestedTensor


class BoxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device('cpu')

    @classmethod
    def tearDown(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    #@unittest.skip('skip fo debug')
    def test_box_cxcyhw_to_xyxy(self):
        box = [120, 60, 40, 50]
        box = paddle.to_tensor(box)

        new_box = box_cxcywh_to_xyxy(box)
        new_box = new_box.numpy().tolist()
        self.assertEqual(new_box, [100, 35, 140, 85])

    #@unittest.skip('skip fo debug')
    def test_box_xyxy_to_cxcyhw(self):
        box = [100, 35, 140, 85]
        box = paddle.to_tensor(box)

        new_box = box_xyxy_to_cxcywh(box)
        new_box = new_box.numpy().tolist()
        self.assertEqual(new_box, [120, 60, 40, 50])

    #@unittest.skip('skip fo debug')
    def test_box_area(self):
        box = [[100, 35, 140, 85], [10, 30, 20, 100]]
        box = paddle.to_tensor(box)
        area = box_area(box)
        self.assertEqual(area[0], 2000)
        self.assertEqual(area[1], 700)

    #@unittest.skip('skip fo debug')
    def test_box_iou(self):
        boxes = [[100, 100, 300, 400],
                 [120, 160, 230, 340],
                 [200, 320, 500, 580],
                 [400, 450, 700, 550],
                 [450, 80, 580, 210]]
        boxes = paddle.to_tensor(boxes).astype('float32')
        iou, union = box_iou(boxes, boxes)
        #print(iou)
        #print(union)

        self.assertEqual(union[0][0], 60000) # area of box1
        self.assertEqual(union[1][1], 19800) # area of box2
        self.assertEqual(union[2][2], 78000) # area of box3
        self.assertEqual(union[3][3], 30000) # area of box4
        self.assertEqual(union[4][4], 16900) # area of box5

        self.assertEqual(union[0][1], 60000) # box2 in box1: res=area of box1
        self.assertEqual(union[0][2], 130000) # area of box1 + box3  - overlap(80*100) 
        self.assertEqual(union[0][3], 90000) # no overlap, area box1 + box4
        self.assertEqual(union[0][4], 76900) # no overlap, area box1 + box5


        self.assertAlmostEqual(iou[0][1], 0.33, 4)
        self.assertAlmostEqual(iou[0][2], 8000/130000, 4)
        self.assertAlmostEqual(iou[0][3], 0, 4)
        self.assertAlmostEqual(iou[0][4], 0, 4)
        

    #@unittest.skip('skip fo debug')
    def test_generalized_box_iou(self):
        boxes = [[100, 100, 300, 400],
                 [120, 160, 230, 340],
                 [200, 320, 500, 580],
                 [400, 450, 700, 550],
                 [450, 80, 580, 210]]
        boxes = paddle.to_tensor(boxes).astype('float32')
        giou = generalized_box_iou(boxes, boxes)
        #print(giou)

        self.assertAlmostEqual(giou[0][0], 1, 4)
        self.assertAlmostEqual(giou[0][1], 0.33, 4)
        self.assertAlmostEqual(giou[0][2].numpy()[0], -0.2613, 3)
        self.assertAlmostEqual(giou[0][3].numpy()[0], -0.6666, 3)
        self.assertAlmostEqual(giou[0][4].numpy()[0], -0.4993, 3)
    
    #@unittest.skip('skip fo debug')
    def test_masks_to_boxes(self):
        masks = paddle.ones([1, 50, 50])
        masks[:, 0:20, :] = 0 
        masks[:, 45::, :] = 0 
        masks[:, :, 0:10] = 0 
        masks[:, :, 49::] = 0 

        boxes = masks_to_boxes(masks)
        self.assertEqual(boxes[0].numpy()[0], 10)
        self.assertEqual(boxes[0].numpy()[1], 20)
        self.assertEqual(boxes[0].numpy()[2], 48)
        self.assertEqual(boxes[0].numpy()[3], 44)
