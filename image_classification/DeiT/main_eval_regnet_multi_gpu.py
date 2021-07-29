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

"""RegNet validation using multiple GPU """

import sys
import os
import time
import logging
import argparse
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from datasets import get_dataloader, get_dataset
from regnet import build_regnet as build_model
from utils import AverageMeter
from config import get_config
from config import update_config


parser = argparse.ArgumentParser('RegNet')
parser.add_argument('-cfg', type=str, default=None)
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-image_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
parser.add_argument('-eval', action='store_true')
arguments = parser.parse_args()


log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")

# get default config
config = get_config()
# update config by arguments
config = update_config(config, arguments)

# set output folder
if not config.EVAL:
    config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
else:
    config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

# set logging format
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')


def validate(dataloader, model, criterion, total_batch, debug_steps=100):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
    Returns:
        val_loss_meter.avg
        val_acc1_meter.avg
        val_acc5_meter.avg
        val_time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()
    time_st = time.time()

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            label = data[1]

            output = model(image)
            loss = criterion(output, label)

            pred = F.softmax(output)
            acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1))
            acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5)

            dist.all_reduce(loss)
            dist.all_reduce(acc1)
            dist.all_reduce(acc5)
            loss = loss / dist.get_world_size()
            acc1 = acc1 / dist.get_world_size()
            acc5 = acc5 / dist.get_world_size()

            batch_size = paddle.to_tensor(image.shape[0])
            dist.all_reduce(batch_size)

            val_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])
            val_acc1_meter.update(acc1.numpy()[0], batch_size.numpy()[0])
            val_acc5_meter.update(acc5.numpy()[0], batch_size.numpy()[0])

            if batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {val_loss_meter.avg:.4f}, " +
                    f"Avg Acc@1: {val_acc1_meter.avg:.4f}, "+
                    f"Avg Acc@5: {val_acc5_meter.avg:.4f}")

    val_time = time.time() - time_st
    return val_loss_meter.avg, val_acc1_meter.avg, val_acc5_meter.avg, val_time


def main_worker(*args):
    # 0. Preparation
    dist.init_parallel_env()
    last_epoch = config.TRAIN.LAST_EPOCH
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    seed = config.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 1. Create model
    model = build_model()
    model = paddle.DataParallel(model)
    # 2. Create train and val dataloader
    dataset_val = args[0]
    dataloader_val = get_dataloader(config, dataset_val, 'test', True)
    total_batch_val = len(dataloader_val)
    logging.info(f'----- Total # of val batch (single gpu): {total_batch_val}')
    # 3. define val criterion
    val_criterion = nn.CrossEntropyLoss()
    # 4. Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")
    # 5. Validation
    if config.EVAL:
        logger.info('----- Start Validating')
        val_loss, val_acc1, val_acc5, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=val_criterion,
            total_batch=total_batch_val,
            debug_steps=config.REPORT_FREQ)
        logger.info(f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Acc@1: {val_acc1:.4f}, " +
                    f"Validation Acc@5: {val_acc5:.4f}, " +
                    f"time: {val_time:.2f}")


def main():
    dataset_val = get_dataset(config, mode='val')
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(dataset_val, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
