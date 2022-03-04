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
from mixup import Mixup
from losses import LabelSmoothingCrossEntropyLoss
from losses import SoftTargetCrossEntropyLoss
from transformer import build_transformer as build_model
from utils import AverageMeter
from utils import WarmupCosineScheduler
from utils import get_exclude_from_weight_decay_fn
from utils import get_params_groups
from utils import cosine_scheduler
from utils import adjust_learning_rate
from utils import interpolate_pos_embed
import lr_decay
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
          optimizer,
          criterion,
          base_lr,
          min_lr,
          epoch,
          warmup_epochs,
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
        optimizer: nn.optimizer
        criterion: nn.XXLoss
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
        train_acc_meter.avg: float, average acc@1 on current process/gpu
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc_meter.avg: float, average acc@1 on all processes/gpus
        train_time: float, training time
    """
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    master_loss_meter = AverageMeter()
    master_acc_meter = AverageMeter()

    time_st = time.time()

    if amp is True:
        scaler = paddle.amp.GradScaler() # default init_loss_scaling = 32768
    optimizer.clear_grad()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        label = data[1]
        label_orig = label.clone()
        batch_size = images.shape[0]

        if mixup_fn is not None:
            images, label = mixup_fn(images, label_orig)

        if batch_id % accum_iter == 0:
            lr = adjust_learning_rate(optimizer,
                                      base_lr,
                                      min_lr,
                                      batch_id / total_batch + epoch - 1, 
                                      warmup_epochs, 
                                      total_epochs)
        # forward
        with paddle.amp.auto_cast(amp is True):
            output = model(images)
            loss = criterion(output, label)

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

        pred = F.softmax(output)
        if mixup_fn:
            acc = paddle.metric.accuracy(pred, label_orig).item()
        else:
            acc = paddle.metric.accuracy(pred, label_orig.unsqueeze(1)).item()

        # sync from other gpus for overall loss and acc

        master_loss = all_reduce_mean(loss_value)
        master_acc = all_reduce_mean(acc)
        master_batch_size = all_reduce_mean(batch_size)

        master_loss_meter.update(master_loss, master_batch_size)
        master_acc_meter.update(master_acc, master_batch_size)
        train_loss_meter.update(loss_value, batch_size)
        train_acc_meter.update(acc, batch_size)

        if batch_id % debug_steps == 0 or batch_id + 1 == len(dataloader):
            local_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                             f"Step[{batch_id:04d}/{total_batch:04d}], " +
                             f"Lr: {optimizer.get_lr():04f}, " +
                             f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f}), " +
                             f"Avg Acc: {train_acc_meter.avg:.4f}")
            master_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                              f"Step[{batch_id:04d}/{total_batch:04d}], " +
                              f"Lr: {optimizer.get_lr():04f}, " +
                              f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f}), " +
                              f"Avg Acc: {master_acc_meter.avg:.4f}")
            write_log(local_logger, master_logger, local_message, master_message)

    train_time = time.time() - time_st
    dist.barrier()
    return (train_loss_meter.avg,
            train_acc_meter.avg,
            master_loss_meter.avg,
            master_acc_meter.avg,
            train_time)


@paddle.no_grad()
def validate(dataloader,
             model,
             criterion,
             total_batch,
             debug_steps=100,
             local_logger=None,
             master_logger=None):
    """Validation for the whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current processes/gpus
        val_acc5_meter.avg: float, average top5 accuracy on current processes/gpus
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc1_meter.avg: float, average top1 accuracy on all processes/gpus
        master_acc5_meter.avg: float, average top5 accuracy on all processes/gpus
        val_time: float, validation time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()
    master_loss_meter = AverageMeter()
    master_acc1_meter = AverageMeter()
    master_acc5_meter = AverageMeter()

    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        label = data[1]
        batch_size = images.shape[0]

        output = model(images)
        loss = criterion(output, label)
        loss_value = loss.item()

        pred = F.softmax(output)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1)).item()
        acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5).item()

        # sync from other gpus for overall loss and acc
        master_loss = all_reduce_mean(loss_value)
        master_acc1 = all_reduce_mean(acc1)
        master_acc5 = all_reduce_mean(acc5)
        master_batch_size = all_reduce_mean(batch_size)

        master_loss_meter.update(master_loss, master_batch_size)
        master_acc1_meter.update(master_acc1, master_batch_size)
        master_acc5_meter.update(master_acc5, master_batch_size)
        val_loss_meter.update(loss_value, batch_size)
        val_acc1_meter.update(acc1, batch_size)
        val_acc5_meter.update(acc5, batch_size)

        if batch_id % debug_steps == 0:
            local_message = (f"Step[{batch_id:04d}/{total_batch:04d}], " +
                             f"Avg Loss: {val_loss_meter.avg:.4f}, " +
                             f"Avg Acc@1: {val_acc1_meter.avg:.4f}, " +
                             f"Avg Acc@5: {val_acc5_meter.avg:.4f}")
            master_message = (f"Step[{batch_id:04d}/{total_batch:04d}], " +
                              f"Avg Loss: {master_loss_meter.avg:.4f}, " + 
                              f"Avg Acc@1: {master_acc1_meter.avg:.4f}, " +
                              f"Avg Acc@5: {master_acc5_meter.avg:.4f}")
            write_log(local_logger, master_logger, local_message, master_message)

    val_time = time.time() - time_st
    dist.barrier()
    return (val_loss_meter.avg,
            val_acc1_meter.avg,
            val_acc5_meter.avg,
            master_loss_meter.avg,
            master_acc1_meter.avg,
            master_acc5_meter.avg,
            val_time)


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

    # STEP 2: Create train and val dataloader
    if not config.EVAL:
        dataset_train = args[1]
        dataloader_train = get_dataloader(config, dataset_train, 'train', True)
        total_batch_train = len(dataloader_train)
        message = f'----- Total # of train batch (single gpu): {total_batch_train}'
        write_log(local_logger, master_logger, message)

    dataset_val = args[2]
    dataloader_val = get_dataloader(config, dataset_val, 'val', True)
    total_batch_val = len(dataloader_val)
    message = f'----- Total # of val batch (single gpu): {total_batch_val}'
    write_log(local_logger, master_logger, message)

    # STEP 3: Define Mixup function
    mixup_fn = None
    if config.TRAIN.MIXUP_PROB > 0 or config.TRAIN.CUTMIX_ALPHA > 0 or config.TRAIN.CUTMIX_MINMAX is not None:
        mixup_fn = Mixup(mixup_alpha=config.TRAIN.MIXUP_ALPHA,
                         cutmix_alpha=config.TRAIN.CUTMIX_ALPHA,
                         cutmix_minmax=config.TRAIN.CUTMIX_MINMAX,
                         prob=config.TRAIN.MIXUP_PROB,
                         switch_prob=config.TRAIN.MIXUP_SWITCH_PROB,
                         mode=config.TRAIN.MIXUP_MODE,
                         label_smoothing=config.TRAIN.SMOOTHING)

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
    # set lr according to batch size and world size (hacked from Swin official code and modified for CSwin)
    if not config.EVAL:
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
            params_groups = lr_decay.param_groups_lrd(
                model=model,
                no_weight_decay_list=['encoder_position_embedding', 'cls_token'],
                weight_decay=config.TRAIN.WEIGHT_DECAY,
                layer_decay=config.TRAIN.LAYER_DECAY)
            optimizer = paddle.optimizer.AdamW(
                parameters=params_groups,
                learning_rate=config.TRAIN.BASE_LR, #scheduler if scheduler is not None else config.TRAIN.BASE_LR,
                beta1=config.TRAIN.OPTIMIZER.BETAS[0],
                beta2=config.TRAIN.OPTIMIZER.BETAS[1],
                weight_decay=config.TRAIN.WEIGHT_DECAY, # set by params_groups, this vaule is not effectitve 
                epsilon=config.TRAIN.OPTIMIZER.EPS,
                grad_clip=clip)
        elif config.TRAIN.OPTIMIZER.NAME == "AdamWDL":
            name_dict = dict()
            wd_exclude_list = ['encoder_position_embedding', 'cls_token']
            for n, p in model.named_parameters():
                # name_dict is for AdamWDL argument 'name_dict'
                name_dict[p.name] = n
                # add no decay param name to weight exclude list, for AramWDL argument 'apply_decay_param_fn' 
                if p.ndim == 1 or n.endswith('.bias'):
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

    # STEP 6: Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED + '.pdparams')

        if not config.EVAL:
            keys = ['encoder.norm.weight', 'encoder.norm.bias',
                    'classfier.weight', 'classifier.bias']
            if config.MODEL.GLOBAL_POOL:
                if keys[0] in model_state:
                    del model_state[keys[0]]
                if keys[1] in model_state:
                    del model_state[keys[1]]
            if keys[2] in model_state:
                del model_state[keys[2]]
            if keys[3] in model_state:
                del model_state[keys[3]]

        # interpolate position embedding
        interpolate_pos_embed(model, model_state)

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
    
    if dist.get_world_size() > 1:
        model = fleet.distributed_model(model)

    # STEP 7: Validation (eval mode)
    if config.EVAL:
        write_log(local_logger, master_logger, f"----- Start Validation")
        val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion_val,
            total_batch=total_batch_val,
            debug_steps=config.REPORT_FREQ,
            local_logger=local_logger,
            master_logger=master_logger)

        local_message = (f"----- Validation: " +
                         f"Validation Loss: {val_loss:.4f}, " +
                         f"Validation Acc@1: {val_acc1:.4f}, " +
                         f"Validation Acc@5: {val_acc5:.4f}, " +
                         f"time: {val_time:.2f}")

        master_message = (f"----- Validation: " +
                         f"Validation Loss: {avg_loss:.4f}, " +
                         f"Validation Acc@1: {avg_acc1:.4f}, " +
                         f"Validation Acc@5: {avg_acc5:.4f}, " +
                         f"time: {val_time:.2f}")
        write_log(local_logger, master_logger, local_message, master_message)
        return

    
    # STEP 7: Start training (train mode)
    write_log(local_logger, master_logger, f"----- Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # train
        write_log(local_logger, master_logger, f"Train epoch {epoch}. LR={optimizer.get_lr():.6e}")

        train_loss, train_acc, avg_loss, avg_acc, train_time = train(
            dataloader=dataloader_train,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            base_lr=config.TRAIN.BASE_LR,
            min_lr=config.TRAIN.END_LR,
            epoch=epoch,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batch=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            mixup_fn=mixup_fn,
            amp=config.AMP,
            local_logger=local_logger,
            master_logger=master_logger)

        local_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                         f"Lr: {optimizer.get_lr():.4f}, " +
                         f"Train Loss: {train_loss:.4f}, " +
                         f"Train Acc: {train_acc:.4f}, " +
                         f"time: {train_time:.2f}")

        master_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                          f"Lr: {optimizer.get_lr():.4f}, " +
                          f"Train Loss: {avg_loss:.4f}, " +
                          f"Train Acc: {avg_acc:.4f}, " +
                          f"time: {train_time:.2f}")
        write_log(local_logger, master_logger, local_message, master_message)

        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            write_log(local_logger, master_logger, f'----- Validation after Epoch: {epoch}')
            val_loss, val_acc1, val_acc5, avg_loss, avg_acc1, avg_acc5, val_time = validate(
                dataloader=dataloader_val,
                model=model,
                criterion=criterion_val,
                total_batch=total_batch_val,
                debug_steps=config.REPORT_FREQ,
                local_logger=local_logger,
                master_logger=master_logger)

            local_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                             f"Validation Loss: {val_loss:.4f}, " +
                             f"Validation Acc@1: {val_acc1:.4f}, " +
                             f"Validation Acc@5: {val_acc5:.4f}, " +
                             f"time: {val_time:.2f}")

            master_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                             f"Validation Loss: {avg_loss:.4f}, " +
                             f"Validation Acc@1: {avg_acc1:.4f}, " +
                             f"Validation Acc@5: {avg_acc5:.4f}, " +
                             f"time: {val_time:.2f}")
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
    if not config.EVAL:
        config.SAVE = '{}/finetuning-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
    else:
        config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)
    # get train dataset if in train mode
    if config.EVAL:
        dataset_train = None
    else:
        dataset_train = get_dataset(config, mode='train')
    # get val dataset
    dataset_val = get_dataset(config, mode='val')
    # start training
    #config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    #dist.spawn(main_worker, args=(config, dataset_train, dataset_val), nprocs=config.NGPUS)
    main_worker(config, dataset_train, dataset_val)


if __name__ == "__main__":
    main()
