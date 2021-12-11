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

import cv2
import numpy as np

def visualize(img_path, pred, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        img_path (str): The path of input image.
        pred (np.ndarray): The predict result of segmentation model.
        weight (float): The image weight of visual image, and the result weight 
        is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): the visualized result.
    """

    color_map = get_pseudo_color_map(256)
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(pred, color_map[:, 0])
    c2 = cv2.LUT(pred, color_map[:, 1])
    c3 = cv2.LUT(pred, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))
    img = cv2.imread(img_path)
    vis_result = cv2.addWeighted(img, weight, pseudo_img, 1 - weight, 0)
    return vis_result

def get_cityscapes_color_map():
    """
    Get the color map of Cityscapes dataset 

    Returns:
        color_map (list): The color map of Cityscapes 
    """
    num_cls = 20
    color_map = [0] * (num_cls * 3)
    color_map[0:3] = (128, 64, 128)       # 0: 'road' 
    color_map[3:6] = (244, 35,232)        # 1 'sidewalk'
    color_map[6:9] = (70, 70, 70)         # 2''building'
    color_map[9:12] = (102,102,156)       # 3 wall
    color_map[12:15] =  (190,153,153)     # 4 fence
    color_map[15:18] = (153,153,153)      # 5 pole
    color_map[18:21] = (250,170, 30)      # 6 'traffic light'
    color_map[21:24] = (220,220, 0)       # 7 'traffic sign'
    color_map[24:27] = (107,142, 35)      # 8 'vegetation'
    color_map[27:30] = (152,251,152)      # 9 'terrain'
    color_map[30:33] = ( 70,130,180)      # 10 sky
    color_map[33:36] = (220, 20, 60)      # 11 person
    color_map[36:39] = (255, 0, 0)        # 12 rider
    color_map[39:42] = (0, 0, 142)        # 13 car
    color_map[42:45] = (0, 0, 70)         # 14 truck
    color_map[45:48] = (0, 60,100)        # 15 bus
    color_map[48:51] = (0, 80,100)        # 16 train
    color_map[51:54] = (0, 0,230)         # 17 'motorcycle'
    color_map[54:57] = (119, 11, 32)      # 18 'bicycle'
    color_map[57:60] = (105, 105, 105)
    return color_map

def get_pseudo_color_map(num_classes=256):
    """
    Get the pseduo color map for visualizing the segmentation mask,

    Args:
        num_classes (int): Number of classes.

    Returns:
        colar_map (list): The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map
