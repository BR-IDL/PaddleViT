#!/usr/bin/python3
import os
import time
import shutil
import random
import argparse
import numpy as np
import cv2
from PIL import Image as PILImage
import shutil
import paddle
import paddle.nn.functional as F
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import *
from src.api import infer
from src.transforms import Compose, Resize, Normalize 
from src.models import get_model
from src.utils import logger, progbar
from src.utils import load_entire_model, resume
from src.utils.vis import visualize
from src.utils.vis import get_cityscapes_color_map

def parse_args():
    parser = argparse.ArgumentParser(description="PaddleViT-Seg Demo")
    parser.add_argument(
        "--config",
        dest='cfg',
        default="../configs/setr/SETR_PUP_Large_768x768_80k_cityscapes_bs_8.yaml", 
        type=str, 
        help="the config file."
    )
    parser.add_argument(
        "--model_path", 
        default="../pretrain_models/setr/SETR_PUP_cityscapes_b8_80k.pdparams", 
        type=str,
        help="the path of weights file (segmentation model)"
    )
    parser.add_argument(
        "--pretrained_backbone",
        default="../pretrain_models/backbones/vit_large_patch16_224.pdparams",
        type=str,
        help="the path of weights file (backbone)"
    )
    parser.add_argument(                                                                                                                                                                                            
        "--img_dir", 
        default="./img/",
        type=str,
        help="the directory of input images"
    )
    parser.add_argument(
        "--results_dir", 
        default="./results/",
        type=str,
        help="the directory of segmentation results"
    )
    return parser.parse_args()

if __name__ == '__main__':
    config = get_config()
    args = parse_args()
    config = update_config(config, args)
    place = 'gpu' if config.VAL.USE_GPU else 'cpu'
    paddle.set_device(place)
    # build model
    model = get_model(config)
    if args.model_path:
        load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')
    model.eval()

    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir)

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    # build transforms for input images
    transforms_val = [ Resize(target_size=config.VAL.IMAGE_BASE_SIZE,
                              keep_ori_size=config.VAL.KEEP_ORI_SIZE),
                       Normalize(mean=config.VAL.MEAN, std=config.VAL.STD)]
    transforms_val = Compose(transforms_val)
    logger.info("Start predicting: ")
    img_files = os.listdir(args.img_dir)
    img_files = [ os.path.join(args.img_dir, item) for item in img_files ]
    print("img_files: ", img_files)
    progbar_val = progbar.Progbar(target=len(img_files), verbose=1)
    with paddle.no_grad():
        for i, img_path in enumerate(img_files):
            img = cv2.imread(img_path)
            ori_shape = img.shape[:2]
            img, _ = transforms_val(img)
            img = img[np.newaxis, ...]
            img = paddle.to_tensor(img)
            pred = infer.ss_inference(
                model=model,
                img=img,
                ori_shape=ori_shape,
                is_slide=True,
                base_size=config.VAL.IMAGE_BASE_SIZE,
                stride_size=config.VAL.STRIDE_SIZE,
                crop_size=config.VAL.CROP_SIZE,
                num_classes=config.DATA.NUM_CLASSES,
                rescale_from_ori=config.VAL.RESCALE_FROM_ORI)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')
            img_name = os.path.basename(img_path)
            # save image+mask
            mask_added_image = visualize(img_path, pred, weight=0.6)
            mask_added_image_path = os.path.join(args.results_dir, img_name)
            cv2.imwrite(mask_added_image_path, mask_added_image)
            # saving color mask
            pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
            color_map = get_cityscapes_color_map()
            pred_mask.putpalette(color_map)
            pred_saved_path = os.path.join(args.results_dir, 
                img_name.rsplit(".")[0] + "_color.png")
            pred_mask.save(pred_saved_path)
            progbar_val.update(i + 1)           
