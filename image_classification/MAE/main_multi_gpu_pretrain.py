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

"""MAE finetuning using multiple GPU """

import sys
import os
import time
import logging
import argparse
import random
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.distributed import fleet
from datasets import get_dataloader
from datasets import get_dataset
from transformer import build_mae_pretrain as build_model
from utils import AverageMeter
from utils import get_exclude_from_weight_decay_fn
from utils import get_params_groups
from utils import adjust_learning_rate
from config import get_config
from config import update_config
import paddlenlp


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


def get_logger(file_path):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    local_rank = dist.get_rank()
    filename = os.path.join(file_path, f'log_all.txt')
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")

    # local_logger for each process/GPU
    local_logger = logging.getLogger(f'local_{local_rank}')
    filename = os.path.join(file_path, f'log_{local_rank}.txt')
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(log_format))
    local_logger.addHandler(fh)
    ## console
    #sh = logging.StreamHandler(sys.stdout)
    #sh.setFormatter(logging.Formatter(log_format))
    #local_logger.addHandler(sh)

    # master_logger records avg performance
    if local_rank == 0:
        master_logger = logging.getLogger('master')
        # log.txt
        filename = os.path.join(file_path, f'log.txt')
        fh = logging.FileHandler(filename)
        fh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(fh)
        # console
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(log_format))
        master_logger.addHandler(sh)
    else:
        master_logger = None
    return local_logger, master_logger


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


def all_reduce_mean(x):
    world_size = dist.get_world_size()
    if world_size > 1:
        x_reduce = paddle.to_tensor(x)
        dist.all_reduce(x_reduce)
        x_reduce = x_reduce / world_size 
        return x_reduce.item()
    else:
        return x


def train(dataloader,
          model,
          mask_ratio,
          optimizer,
          base_lr,
          min_lr,
          epoch,
          warmup_epochs,
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
        masek_ratio, float, mask ratio
        optimizer: nn.optimizer
        base_lr: float, base learning rate
        min_lr: float, minimum lr
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

    time_st = time.time()

    if amp is True:
        scaler = paddle.amp.GradScaler() # default init_loss_scaling = 32768
    optimizer.clear_grad()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        batch_size = images.shape[0]
        # adjust learning rate
        if batch_id % accum_iter == 0:
            adjust_learning_rate(optimizer,
                                 base_lr,
                                 min_lr,
                                 batch_id / total_batch + epoch - 1, 
                                 warmup_epochs, 
                                 total_epochs)
        # forward
        with paddle.amp.auto_cast(amp is True):
            loss, _, _ = model(images)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        if not amp: # fp32
            loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()
        else:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                # amp for param group refer here: https://github.com/PaddlePaddle/Paddle/issues/37188
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()

        # sync from other gpus for overall loss and acc
        master_loss = all_reduce_mean(loss_value)
        master_batch_size = all_reduce_mean(batch_size)
        master_loss_meter.update(master_loss, master_batch_size)
        train_loss_meter.update(loss_value, batch_size)
        if batch_id % debug_steps == 0 or batch_id + 1 == len(dataloader):
            local_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                             f"Step[{batch_id:04d}/{total_batch:04d}], " +
                             f"Lr: {optimizer.get_lr():04f}, " +
                             f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f})")
            master_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                              f"Step[{batch_id:04d}/{total_batch:04d}], " +
                              f"Lr: {optimizer.get_lr():04f}, " +
                              f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f})")
            write_log(local_logger, master_logger, local_message, master_message)

    train_time = time.time() - time_st
    dist.barrier()
    return train_loss_meter.avg, master_loss_meter.avg, train_time


def main_worker(*args):
    # STEP 0: Preparation
    #dist.init_parallel_env()
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    config = args[0]
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # logger for each process/gpu
    local_logger, master_logger = get_logger(config.SAVE)
    message = f'----- world_size = {world_size}, local_rank = {local_rank}'
    write_log(local_logger, master_logger, message)
    
    # STEP 1: Create model
    paddle.device.set_device('gpu')
    model = build_model(config)
    if dist.get_world_size() > 1:
        strategy = fleet.DistributedStrategy()
        ## Hybrid Parallel Training
        strategy.hybrid_configs = {}
        fleet.init(is_collective=True, strategy=strategy)

    # STEP 2: Create train dataloader
    dataset_train = args[1]
    dataloader_train = get_dataloader(config, dataset_train, 'train', True)
    total_batch_train = len(dataloader_train)
    message = f'----- Total # of train batch (single gpu): {total_batch_train}'
    write_log(local_logger, master_logger, message)

    # STEP 3: Define optimizer and lr_scheduler
    # set lr according to batch size and world size (hacked from Swin official code and modified for CSwin)
    if config.TRAIN.LINEAR_SCALED_LR is not None:
        effective_batch_size = config.DATA.BATCH_SIZE * config.TRAIN.ACCUM_ITER * world_size            
        config.TRAIN.BASE_LR = (
            config.TRAIN.BASE_LR * effective_batch_size / config.TRAIN.LINEAR_SCALED_LR
        )
        write_log(local_logger, master_logger, f'Base lr is scaled to: {config.TRAIN.BASE_LR}')
        

    if config.TRAIN.GRAD_CLIP:
        clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    else:
        clip = None

    if config.TRAIN.OPTIMIZER.NAME == "AdamW":
        #wd_exclude_list = ['encoder_position_embedding', 'cls_token']
        wd_exclude_list = []
        for n, p in model.named_parameters():
            if p.stop_gradient is True:
                continue
            if len(p.shape) == 1 or n.endswith('.bias'):
                wd_exclude_list.append(n)
        print('no_decay param names: ', wd_exclude_list)
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=config.TRAIN.BASE_LR, #scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY, # set by params_groups, this vaule is not effectitve 
            apply_decay_param_fun=get_exclude_from_weight_decay_fn(wd_exclude_list),
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "AdamWDL":
        name_dict = dict()
        wd_exclude_list = ['encoder_position_embedding', 'cls_token']
        for n, p in model.named_parameters():
            # name_dict is for AdamWDL argument 'name_dict'
            name_dict[p.name] = n
            # add no decay param name to weight exclude list, for AramWDL argument 'apply_decay_param_fn' 
            if p.stop_gradient is True:
                continue
            if len(p.shape) == 1 or n.endswith('.bias'):
                wd_exclude_list.append(n)
        print('no_decay param names: ', wd_exclude_list)

        optimizer = paddlenlp.ops.optimizer.AdamWDL(
            learning_rate=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            layerwise_decay=config.TRAIN.LAYER_DECAY,
            n_layers=config.MODEL.TRANS.ENCODER.DEPTH,
            set_param_lr_fun=lr_decay.lr_setting,
            parameters=model.parameters(),
            name_dict=name_dict,
            apply_decay_param_fun=get_exclude_from_weight_decay_fn(wd_exclude_list),
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip)
    else:
        message = f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}."
        write_log(local_logger, master_logger, message, None, 'fatal')
        raise NotImplementedError(message)

    # STEP 4: Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED + '.pdparams')
        model.set_state_dict(model_state)
        message = f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}.pdparams"
        write_log(local_logger, master_logger, message)

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME+'.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME+'.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        message = f"----- Resume Training: Load model and optmizer from {config.MODEL.RESUME}.pdparams/.pdopts"
        write_log(local_logger, master_logger, message)
        if config.TRAIN.LAST_EPOCH == -1:
            message = f"----- Resume Training: LAST_EPOCH should not be [-1]"
            write_log(local_logger, master_logger, message, None, 'fatal')
    
    if dist.get_world_size() > 1:
        model = fleet.distributed_model(model)

    # STEP 5: Start training (train mode)
    write_log(local_logger, master_logger, f"----- Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # train
        write_log(local_logger, master_logger, f"Train epoch {epoch}. LR={optimizer.get_lr():.6e}")

        train_loss, avg_loss, train_time = train(
            dataloader=dataloader_train,
            model=model,
            mask_ratio=config.MODEL.TRANS.MASK_RATIO,
            optimizer=optimizer,
            base_lr=config.TRAIN.BASE_LR,
            min_lr=config.TRAIN.END_LR,
            epoch=epoch,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batch=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            amp=config.AMP,
            local_logger=local_logger,
            master_logger=master_logger)

        local_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                         f"Lr: {optimizer.get_lr():.4f}, " +
                         f"Train Loss: {train_loss:.4f}, " +
                         f"time: {train_time:.2f}")

        master_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                          f"Lr: {optimizer.get_lr():.4f}, " +
                          f"Train Loss: {avg_loss:.4f}, " +
                          f"time: {train_time:.2f}")
        write_log(local_logger, master_logger, local_message, master_message)

        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{avg_loss}")
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
    config.SAVE = '{}/finetuning-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)
    # get dataset
    dataset_train = get_dataset(config, mode='train')
    # start training
    #config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    #dist.spawn(main_worker, args=(config, dataset_train, ), nprocs=config.NGPUS)
    main_worker(config, dataset_train, )


if __name__ == "__main__":
    main()
