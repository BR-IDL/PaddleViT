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

import os
import time
import random
import argparse
import numpy as np
from collections import deque
import paddle
import paddle.nn as nn
from config import *
from src.utils import logger
from src.datasets import get_dataset
from src.models import get_model
from src.transforms import *
from src.utils import TimeAverager, calculate_eta, resume, get_dataloader
from src.models.solver import get_scheduler, get_optimizer
from src.models.losses import get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual Transformer for semantic segmentation')
    parser.add_argument(
        "--config", 
        dest='cfg',
        default=None, 
        type=str,
        help="The config file."
    )
    return parser.parse_args()

def main():
    config = get_config()
    args = parse_args()
    config = update_config(config, args)
    place = 'gpu' if config.TRAIN.USE_GPU else 'cpu'
    paddle.set_device(place)
    # build  model
    model = get_model(config)
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    # build scheduler
    lr_scheduler = get_scheduler(config)
    # build optimizer
    optimizer = get_optimizer(model, lr_scheduler, config)
    # bulid train transforms
    transforms_train = get_transforms(config)
    # build dataset_train
    dataset_train = get_dataset(config, data_transform=transforms_train, mode='train')
    train_loader = get_dataloader(dataset=dataset_train,
                                  shuffle=True,
                                  batch_size=config.DATA.BATCH_SIZE,
                                  num_iters=config.TRAIN.ITERS,
                                  num_workers=config.DATA.NUM_WORKERS)
    # build loss function
    loss_func = get_loss_function(config)
    # TODO(wutianyiRosun@gmail.com): Resume from checkpoints, and update start_iter

    # build workspace for saving checkpoints
    if not os.path.isdir(config.SAVE_DIR):
        if os.path.exists(config.SAVE_DIR):
            os.remove(config.SAVE_DIR)
        os.makedirs(config.SAVE_DIR)
    logger.info("train_loader.len= {}".format(len(train_loader)))
    start_iter = 0
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
            logger.info("using dist training")
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)
    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(dataset_train) // config.DATA.BATCH_SIZE
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    cur_iter = start_iter
    # begin training
    for data in train_loader:
        cur_iter += 1
        reader_cost_averager.record(time.time() - batch_start)
        images = data[0]
        labels = data[1].astype('int64')
        if nranks > 1:
            logits_list = ddp_model(images)
        else:
            logits_list = model(images)
        loss_list = loss_func(logits_list, labels)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        lr = optimizer.get_lr()
        if isinstance(optimizer._learning_rate,paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()
        model.clear_gradients()
        avg_loss += loss.numpy()[0]
        if not avg_loss_list:
            avg_loss_list = [l.numpy() for l in loss_list]
        else:
            for i in range(len(loss_list)):
                avg_loss_list[i] += loss_list[i].numpy()
        batch_cost_averager.record(
            time.time() - batch_start, num_samples=config.DATA.BATCH_SIZE)
        if (cur_iter) % config.LOGGING_INFO_FREQ == 0 and local_rank == 0:
            avg_loss /= config.LOGGING_INFO_FREQ
            avg_loss_list = [l[0] / config.LOGGING_INFO_FREQ for l in avg_loss_list]
            remain_iters = config.TRAIN.ITERS - cur_iter
            avg_train_batch_cost = batch_cost_averager.get_average()
            avg_train_reader_cost = reader_cost_averager.get_average()
            eta = calculate_eta(remain_iters, avg_train_batch_cost)
            logger.info("[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.8f}, batch_cost:\
                {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}".format(
                (cur_iter - 1) // iters_per_epoch + 1, cur_iter, config.TRAIN.ITERS, avg_loss, 
                lr, avg_train_batch_cost, avg_train_reader_cost, 
                batch_cost_averager.get_ips_average(), eta))
            avg_loss = 0.0
            avg_loss_list = []
            reader_cost_averager.reset()
            batch_cost_averager.reset()

        if (cur_iter % config.SAVE_FREQ_CHECKPOINT == 0 or cur_iter == config.TRAIN.ITERS) and local_rank == 0:
            current_save_weigth_file = os.path.join(config.SAVE_DIR,
                "iter_{}_model_state.pdparams".format(cur_iter))
            current_save_opt_file = os.path.join(config.SAVE_DIR,
                "iter_{}_opt_state.pdopt".format(cur_iter))
            paddle.save(model.state_dict(), current_save_weigth_file)
            paddle.save(optimizer.state_dict(), current_save_opt_file)
            save_models.append([current_save_weigth_file,
                                current_save_opt_file])
            logger.info("saving the weights of model to {}".format(
                current_save_weigth_file))
            if len(save_models) > config.KEEP_CHECKPOINT_MAX > 0:
                files_to_remove = save_models.popleft()
                os.remove(files_to_remove[0])
                os.remove(files_to_remove[1])
        batch_start = time.time()
    time.sleep(1.0)

if __name__ == '__main__':
    main()
