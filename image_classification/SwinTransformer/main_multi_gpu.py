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

"""Swin training/validation using multiple GPU """

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
from utils import AverageMeter
from utils import WarmupCosineScheduler
from utils import get_exclude_from_weight_decay_fn
from config import get_config
from config import update_config
from mixup import Mixup
from losses import LabelSmoothingCrossEntropyLoss
from losses import SoftTargetCrossEntropyLoss
from losses import DistillationLoss
from swin_transformer import build_swin as build_model


def get_arguments():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('Swin')
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


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          mixup_fn=None,
          amp=False,
          local_logger=None,
          master_logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        mixup_fn: Mixup, mixup instance, default: None
        amp: bool, if True, use mix precision training, default: False
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average top1 accuracy on current process/gpu
        master_train_loss_meter.avg: float, average loss on all processes/gpus
        master_train_acc_meter.avg: float, average top1 accuracy on all processes/gpus
        train_time: float, training time
    """
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    master_train_loss_meter = AverageMeter()
    master_train_acc_meter = AverageMeter()

    if amp is True:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]
        label_orig = label.clone()

        if mixup_fn is not None:
            image, label = mixup_fn(image, label_orig)
        
        if amp is True: # mixed precision training
            with paddle.amp.auto_cast():
                output = model(image)
                loss = criterion(image, output, label)
            scaled = scaler.scale(loss)
            scaled.backward()
            if ((batch_id +1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
        else: # full precision training
            output = model(image)
            loss = criterion(output, label)
            #NOTE: division may be needed depending on the loss function
            # Here no division is needed:
            # default 'reduction' param in nn.CrossEntropyLoss is set to 'mean'
            #loss =  loss / accum_iter
            loss.backward()

            if ((batch_id +1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()

        pred = F.softmax(output)
        if mixup_fn:
            acc = paddle.metric.accuracy(pred, label_orig)
        else:
            acc = paddle.metric.accuracy(pred, label_orig.unsqueeze(1))

        batch_size = paddle.to_tensor(image.shape[0])

        # sync from other gpus for overall loss and acc
        master_loss = loss.clone()
        master_acc = acc.clone()
        master_batch_size = batch_size.clone()
        dist.all_reduce(master_loss)
        dist.all_reduce(master_acc)
        dist.all_reduce(master_batch_size)
        master_loss = master_loss / dist.get_world_size()
        master_acc = master_acc / dist.get_world_size()
        master_train_loss_meter.update(master_loss.numpy()[0], master_batch_size.numpy()[0])
        master_train_acc_meter.update(master_acc.numpy()[0], master_batch_size.numpy()[0])

        train_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])
        train_acc_meter.update(acc.numpy()[0], batch_size.numpy()[0])

        if batch_id % debug_steps == 0:
            if local_logger:
                local_logger.info(
                    f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                    f"Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {train_loss_meter.avg:.4f}, " +
                    f"Avg Acc: {train_acc_meter.avg:.4f}")
            if master_logger and dist.get_rank() == 0:
                master_logger.info(
                    f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                    f"Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg Loss: {master_train_loss_meter.avg:.4f}, " +
                    f"Avg Acc: {master_train_acc_meter.avg:.4f}")

    train_time = time.time() - time_st
    return (train_loss_meter.avg,
            train_acc_meter.avg,
            master_train_loss_meter.avg,
            master_train_acc_meter.avg,
            train_time)


def validate(dataloader,
             model,
             criterion,
             total_batch,
             debug_steps=100,
             local_logger=None,
             master_logger=None):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current process/gpu
        val_acc5_meter.avg: float, average top5 accuracy on current process/gpu
        master_val_loss_meter.avg: float, average loss on all processes/gpus
        master_val_acc1_meter.avg: float, average top1 accuracy on all processes/gpus
        master_val_acc5_meter.avg: float, average top5 accuracy on all processes/gpus
        val_time: float, validation time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()
    master_val_loss_meter = AverageMeter()
    master_val_acc1_meter = AverageMeter()
    master_val_acc5_meter = AverageMeter()
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

            batch_size = paddle.to_tensor(image.shape[0])

            master_loss = loss.clone()
            master_acc1 = acc1.clone()
            master_acc5 = acc5.clone()
            master_batch_size = batch_size.clone()

            dist.all_reduce(master_loss)
            dist.all_reduce(master_acc1)
            dist.all_reduce(master_acc5)
            dist.all_reduce(master_batch_size)
            master_loss = master_loss / dist.get_world_size()
            master_acc1 = master_acc1 / dist.get_world_size()
            master_acc5 = master_acc5 / dist.get_world_size()

            master_val_loss_meter.update(master_loss.numpy()[0], master_batch_size.numpy()[0])
            master_val_acc1_meter.update(master_acc1.numpy()[0], master_batch_size.numpy()[0])
            master_val_acc5_meter.update(master_acc5.numpy()[0], master_batch_size.numpy()[0])

            val_loss_meter.update(loss.numpy()[0], batch_size.numpy()[0])
            val_acc1_meter.update(acc1.numpy()[0], batch_size.numpy()[0])
            val_acc5_meter.update(acc5.numpy()[0], batch_size.numpy()[0])

            if batch_id % debug_steps == 0:
                if local_logger:
                    local_logger.info(
                        f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                        f"Avg Loss: {val_loss_meter.avg:.4f}, " +
                        f"Avg Acc@1: {val_acc1_meter.avg:.4f}, " +
                        f"Avg Acc@5: {val_acc5_meter.avg:.4f}")
                if master_logger and dist.get_rank() == 0:
                    master_logger.info(
                        f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                        f"Avg Loss: {master_val_loss_meter.avg:.4f}, " +
                        f"Avg Acc@1: {master_val_acc1_meter.avg:.4f}, " +
                        f"Avg Acc@5: {master_val_acc5_meter.avg:.4f}")
    val_time = time.time() - time_st
    return (val_loss_meter.avg,
            val_acc1_meter.avg,
            val_acc5_meter.avg,
            master_val_loss_meter.avg,
            master_val_acc1_meter.avg,
            master_val_acc5_meter.avg,
            val_time)


def main_worker(*args):
    # STEP 0: Preparation
    config = args[0]
    dist.init_parallel_env()
    last_epoch = config.TRAIN.LAST_EPOCH
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
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
    local_logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    if local_rank == 0:
        master_logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
    
    # STEP 1: Create model
    model = build_model(config)
    model = paddle.DataParallel(model)

    # STEP 2: Create train and val dataloader
    dataset_train, dataset_val = args[1], args[2]
    # Create training dataloader
    if not config.EVAL:
        dataloader_train = get_dataloader(config, dataset_train, 'train', True)
        total_batch_train = len(dataloader_train)
        local_logger.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
        if local_rank == 0:
            master_logger.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
    # Create validation dataloader
    dataloader_val = get_dataloader(config, dataset_val, 'test', True)
    total_batch_val = len(dataloader_val)
    local_logger.info(f'----- Total # of val batch (single gpu): {total_batch_val}')
    if local_rank == 0:
        master_logger.info(f'----- Total # of val batch (single gpu): {total_batch_val}')

    # STEP 3: Define Mixup function
    mixup_fn = None
    if config.TRAIN.MIXUP_PROB > 0 or config.TRAIN.CUTMIX_ALPHA > 0 or config.TRAIN.CUTMIX_MINMAX is not None:
        mixup_fn = Mixup(mixup_alpha=config.TRAIN.MIXUP_ALPHA,
                         cutmix_alpha=config.TRAIN.CUTMIX_ALPHA,
                         cutmix_minmax=config.TRAIN.CUTMIX_MINMAX,
                         prob=config.TRAIN.MIXUP_PROB,
                         switch_prob=config.TRAIN.MIXUP_SWITCH_PROB,
                         mode=config.TRAIN.MIXUP_MODE,
                         label_smoothing=config.TRAIN.SMOOTHING,
                         num_classes=config.MODEL.NUM_CLASSES)

    # STEP 4: Define criterion
    if config.TRAIN.MIXUP_PROB > 0.:
        criterion = SoftTargetCrossEntropyLoss()
    elif config.TRAIN.SMOOTHING:
        criterion = LabelSmoothingCrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    # only use cross entropy for val
    criterion_val = nn.CrossEntropyLoss()

    # STEP 5: Define optimizer and lr_scheduler
    # set lr according to batch size and world size (hacked from official code)
    linear_scaled_lr = (config.TRAIN.BASE_LR *
        config.DATA.BATCH_SIZE * dist.get_world_size()) / 512.0
    linear_scaled_warmup_start_lr = (config.TRAIN.WARMUP_START_LR *
        config.DATA.BATCH_SIZE * dist.get_world_size()) / 512.0
    linear_scaled_end_lr = (config.TRAIN.END_LR *
        config.DATA.BATCH_SIZE * dist.get_world_size()) / 512.0

    if config.TRAIN.ACCUM_ITER > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUM_ITER
        linear_scaled_warmup_start_lr = linear_scaled_warmup_start_lr * config.TRAIN.ACCUM_ITER
        linear_scaled_end_lr = linear_scaled_end_lr * config.TRAIN.ACCUM_ITER
    
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_START_LR = linear_scaled_warmup_start_lr
    config.TRAIN.END_LR = linear_scaled_end_lr

    scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == "warmupcosine":
        scheduler = WarmupCosineScheduler(learning_rate=config.TRAIN.BASE_LR,
                                          warmup_start_lr=config.TRAIN.WARMUP_START_LR,
                                          start_lr=config.TRAIN.BASE_LR,
                                          end_lr=config.TRAIN.END_LR,
                                          warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
                                          total_epochs=config.TRAIN.NUM_EPOCHS,
                                          last_epoch=config.TRAIN.LAST_EPOCH,
                                          )
    elif config.TRAIN.LR_SCHEDULER.NAME == "cosine":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=config.TRAIN.BASE_LR,
                                                             T_max=config.TRAIN.NUM_EPOCHS,
                                                             last_epoch=last_epoch)
    elif config.scheduler == "multi-step":
        milestones = [int(v.strip()) for v in config.TRAIN.LR_SCHEDULER.MILESTONES.split(",")]
        scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.TRAIN.BASE_LR,
                                                       milestones=milestones,
                                                       gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                       last_epoch=last_epoch)
    else:
        local_logger.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        if local_rank == 0:
            master_logger.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        if config.TRAIN.GRAD_CLIP:
            clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
        else:
            clip = None
        optimizer = paddle.optimizer.Momentum(
            parameters=model.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "AdamW":
        if config.TRAIN.GRAD_CLIP:
            clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
        else:
            clip = None
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip,
            apply_decay_param_fun=get_exclude_from_weight_decay_fn([
                'absolute_pos_embed', 'relative_position_bias_table']),
            )
    else:
        local_logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        if local_rank == 0:
            master_logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # STEP 6: Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        local_logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")
        if local_rank == 0:
            master_logger.info(
                f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME+'.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME+'.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        local_logger.info(
            f"----- Resume Training: Load model and optmizer from {config.MODEL.RESUME}")
        if local_rank == 0:
            master_logger.info(
                f"----- Resume Training: Load model and optmizer from {config.MODEL.RESUME}")
    
    # STEP 7: Validation (eval mode)
    if config.EVAL:
        local_logger.info('----- Start Validating')
        if local_rank == 0:
            master_logger.info('----- Start Validating')
        val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion_val,
            total_batch=total_batch_val,
            debug_steps=config.REPORT_FREQ,
            local_logger=local_logger,
            master_logger=master_logger)
        local_logger.info(f"Validation Loss: {val_loss:.4f}, " +
                          f"Validation Acc@1: {val_acc1:.4f}, " +
                          f"Validation Acc@5: {val_acc5:.4f}, " +
                          f"time: {val_time:.2f}")
        if local_rank == 0:
            master_logger.info(f"Validation Loss: {avg_loss:.4f}, " +
                               f"Validation Acc@1: {avg_acc1:.4f}, " +
                               f"Validation Acc@5: {avg_acc5:.4f}, " +
                               f"time: {val_time:.2f}")
        return

    # STEP 8: Start training and validation (train mode)
    local_logger.info(f"Start training from epoch {last_epoch+1}.")
    if local_rank == 0:
        master_logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        local_logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        if local_rank == 0:
            master_logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_acc, avg_loss, avg_acc, train_time = train(
            dataloader=dataloader_train,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batch=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            mixup_fn=mixup_fn,
            amp=config.AMP,
            local_logger=local_logger,
            master_logger=master_logger)

        scheduler.step()

        local_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                          f"Train Loss: {train_loss:.4f}, " +
                          f"Train Acc: {train_acc:.4f}, " +
                          f"time: {train_time:.2f}")
        if local_rank == 0:
            master_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                               f"Train Loss: {avg_loss:.4f}, " +
                               f"Train Acc: {avg_acc:.4f}, " +
                               f"time: {train_time:.2f}")

        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            local_logger.info(f'----- Validation after Epoch: {epoch}')
            if local_rank == 0:
                master_logger.info(f'----- Validation after Epoch: {epoch}')
            val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                dataloader=dataloader_val,
                model=model,
                criterion=criterion_val,
                total_batch=total_batch_val,
                debug_steps=config.REPORT_FREQ,
                local_logger=local_logger,
                master_logger=master_logger)
            local_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                              f"Validation Loss: {val_loss:.4f}, " +
                              f"Validation Acc@1: {val_acc1:.4f}, " +
                              f"Validation Acc@5: {val_acc5:.4f}, " +
                              f"time: {val_time:.2f}")
            if local_rank == 0:
                master_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                                   f"Validation Loss: {avg_loss:.4f}, " +
                                   f"Validation Acc@1: {avg_acc1:.4f}, " +
                                   f"Validation Acc@5: {avg_acc5:.4f}, " +
                                   f"time: {val_time:.2f}")
        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path + '.pdparams')
                paddle.save(optimizer.state_dict(), model_path + '.pdopt')
                master_logger.info(f"----- Save model: {model_path}.pdparams")
                master_logger.info(f"----- Save optim: {model_path}.pdopt")


def main():
    # config is updated by: (1) config.py, (2) yaml file, (3) arguments
    arguments = get_arguments()
    config = get_config()
    config = update_config(config, arguments)

    # set output folder
    if not config.EVAL:
        config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
    else:
        config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)

    # get dataset and start DDP
    if not config.EVAL:
        dataset_train = get_dataset(config, mode='train')
    else:
        dataset_train = None
    dataset_val = get_dataset(config, mode='val')
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(config, dataset_train, dataset_val, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
