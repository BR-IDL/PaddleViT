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

"""MAE pre-training using multiple GPU """

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
from datasets import get_dataloader
from datasets import get_dataset
from transformer import build_mae_pretrain as build_model
from utils import AverageMeter
from utils import WarmupCosineScheduler
from utils import get_exclude_from_weight_decay_fn
from utils import get_params_groups
from utils import cosine_scheduler
from config import get_config
from config import update_config


def get_arguments():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('MAE')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-amp', action='store_true')
    arguments = parser.parse_args()
    return arguments


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger


def write_log(local_logger, master_logger, msg_local, msg_master=None, level='info'):
    if local_logger:
        if level == 'info':
            local_logger.info(msg_local)
        elif level == 'fatal':
            local_logger.fatal(msg_local)
        else:
            raise ValueError("level must in ['info', 'fatal']")
    if master_logger and dist.get_rank() == 0:
        if msg_master is None:
            msg_master = msg_local
        if level == 'info':
            master_logger.info("MASTER_LOG " + msg_master)
        elif level == 'fatal':
            master_logger.fatal("MASTER_LOG " + msg_master)
        else:
            raise ValueError("level must in ['info', 'fatal']")


def train(dataloader,
          model,
          mask_ratio,
          optimizer,
          lr_schedule,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          amp=False,
          local_logger=None,
          master_logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        mask_ratio: float, percentage of masking patches 
        optimizer: nn.optimizer
        lr_schedule: list of float, lr schdeule
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        amp: bool, if True, use mix precision training, default: False
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        master_loss_meter.avg: float, average loss on all processes/gpus
        train_time: float, training time
    """
    model.train()
    train_loss_meter = AverageMeter()
    master_loss_meter = AverageMeter()

    if amp is True:
        scaler = paddle.amp.GradScaler() # default init_loss_scaling = 32768
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        # set per iteration lr using scheduler
        global_train_iter = total_batch * (epoch - 1) + batch_id # epoch starts from 1
        optimizer.set_lr(lr_schedule[global_train_iter])
        # forward
        with paddle.amp.auto_cast(amp is True):
            loss, _, _ = model(images, mask_ratio)

        if not amp: # fp32
            loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()
        else:
            scaled = scaler.scale(loss)
            scaled.backward()
            # todo: check if manually unscale and clip grad is required here
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                # amp for param group refer here: https://github.com/PaddlePaddle/Paddle/issues/37188
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()

        # sync from other gpus for overall loss and acc
        batch_size = paddle.to_tensor(images.shape[0])
        master_loss = paddle.to_tensor(loss.numpy())
        master_batch_size = paddle.to_tensor(batch_size.numpy())
        dist.all_reduce(master_loss)
        dist.all_reduce(master_batch_size)
        master_loss = master_loss / dist.get_world_size()
        master_loss_meter.update(master_loss.numpy()[0], master_batch_size.numpy()[0])
        train_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])

        if batch_id % debug_steps == 0:
            local_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                             f"Step[{batch_id:04d}/{total_batch:04d}], " +
                             f"LR: {optimizer.get_lr():.6e}, " +
                             f"Avg Loss: {train_loss_meter.avg:.4f}")
            master_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                              f"Step[{batch_id:04d}/{total_batch:04d}], " +
                              f"LR: {optimizer.get_lr():.6e}, " +
                              f"Avg Loss: {master_loss_meter.avg:.4f}")
            write_log(local_logger, master_logger, local_message, master_message)

    train_time = time.time() - time_st
    return train_loss_meter.avg, master_loss_meter.avg, train_time


def main_worker(*args):
    # STEP 0: Preparation
    dist.init_parallel_env()
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    config = args[0]
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # logger for each process/gpu
    local_logger = get_logger(
            filename=os.path.join(config.SAVE, 'log_{}.txt'.format(local_rank)),
            logger_name='local_logger')
    # overall logger
    if local_rank == 0:
        master_logger = get_logger(
            filename=os.path.join(config.SAVE, 'log.txt'),
            logger_name='master_logger')
        master_logger.info(f'\n{config}')
    else:
        master_logger = None

    message = f'----- world_size = {world_size}, local_rank = {local_rank}'
    write_log(local_logger, master_logger, message)
    
    # STEP 1: Create model
    model = build_model(config)
    model = paddle.DataParallel(model)

    # STEP 2: Create train and val dataloader
    dataset_train = args[1]
    dataloader_train = get_dataloader(config, dataset_train, 'train', True)
    total_batch_train = len(dataloader_train)
    message = f'----- Total # of train batch (single gpu): {total_batch_train}'
    write_log(local_logger, master_logger, message)

    # STEP 3: Define criterion: loss is defined in model
    #criterion = nn.MSELoss()

    # STEP 4: Define optimizer and lr_scheduler
    # set lr according to batch size and world size (hacked from Swin official code and modified for CSwin)
    if config.TRAIN.LINEAR_SCALED_LR is not None:
        linear_scaled_lr = (
            config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * world_size) / config.TRAIN.LINEAR_SCALED_LR
        #linear_scaled_warmup_start_lr = (
        #    config.TRAIN.WARMUP_START_LR * config.DATA.BATCH_SIZE * world_size) / config.TRAIN.LINEAR_SCALED_LR
        #linear_scaled_end_lr = (
        #    config.TRAIN.END_LR * config.DATA.BATCH_SIZE * world_size) / config.TRAIN.LINEAR_SCALED_LR
    
        if config.TRAIN.ACCUM_ITER > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUM_ITER
            #linear_scaled_warmup_start_lr = linear_scaled_warmup_start_lr * config.TRAIN.ACCUM_ITER
            #linear_scaled_end_lr = linear_scaled_end_lr * config.TRAIN.ACCUM_ITER
        
        config.TRAIN.BASE_LR = linear_scaled_lr
        #config.TRAIN.WARMUP_START_LR = linear_scaled_warmup_start_lr
        #config.TRAIN.END_LR = linear_scaled_end_lr

    lr_schedule = cosine_scheduler(config.TRAIN.BASE_LR, # add linear scale
                                   config.TRAIN.END_LR,
                                   config.TRAIN.NUM_EPOCHS,
                                   len(dataloader_train),
                                   warmup_epochs=config.TRAIN.WARMUP_EPOCHS)

    params_groups = get_params_groups(model)

    if config.TRAIN.GRAD_CLIP:
        clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    else:
        clip = None

    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=params_groups,
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=params_groups,
            learning_rate=0.0, #scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip)
    else:
        message = f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}."
        write_log(local_logger, master_logger, message, None, 'fatal')
        raise NotImplementedError(message)

    # STEP 5: Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        message = f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}"
        write_log(local_logger, master_logger, message)

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME+'.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME+'.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        message = f"----- Resume Training: Load model and optmizer from {config.MODEL.RESUME}"
        write_log(local_logger, master_logger, message)
        if config.TRAIN.LAST_EPOCH == -1:
            message = f"----- Resume Training: LAST_EPOCH should not be [-1]"
            write_log(local_logger, master_logger, message, None, 'fatal')
    
    # STEP 6: Start training (train mode)
    write_log(local_logger, master_logger, f"----- Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # train
        write_log(local_logger, master_logger, f"Train epoch {epoch}. LR={optimizer.get_lr():.6e}")

        train_loss, avg_loss, train_time = train(
            dataloader=dataloader_train,
            model=model,
            mask_ratio=config.MODEL.TRANS.MASK_RATIO,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            epoch=epoch,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batch=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            amp=config.AMP,
            local_logger=local_logger,
            master_logger=master_logger)

        local_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                         f"Train Loss: {train_loss:.4f}, " +
                         f"time: {train_time:.2f}")

        master_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                          f"Train Loss: {avg_loss:.4f}, " +
                          f"time: {train_time:.2f}")
        write_log(local_logger, master_logger, local_message, master_message)

        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path + '.pdparams')
                paddle.save(optimizer.state_dict(), model_path + '.pdopt')
                message = (f"----- Save model: {model_path}.pdparams \n" +
                           f"----- Save optim: {model_path}.pdopt")
                write_log(local_logger, master_logger, message)


def main():
    # config is updated by: (1) config.py, (2) yaml file, (3) arguments
    arguments = get_arguments()
    config = get_config()
    config = update_config(config, arguments)
    # set output folder
    config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)
    # get dataset
    dataset_train = get_dataset(config, mode='train')
    # start training
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(config, dataset_train, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
