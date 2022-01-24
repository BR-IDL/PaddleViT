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

"""DETR training/validation using multiple GPU """

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
from coco import build_coco
from coco import get_dataloader
from coco_eval import CocoEvaluator
from utils import AverageMeter
from utils import WarmupCosineScheduler
from config import get_config
from config import update_config
from detr import build_detr


def get_arguments():
    """return arguments, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('DETR')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-backbone', type=str, default=None)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
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
          postprocessors,
          base_ds,
          optimizer,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          local_logger=None,
          master_logger=None):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, DETR model
        criterion: nn.Layer
        postprocessors: nn.Layer
        base_ds: coco api for generate CocoEvaluator, pycocotools.coco.COCO(anno_file)
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        train_loss_ce_meter.avg: float, average ce loss on current process/gpu
        train_loss_bbox_meter.avg: float, average bbox loss on current process/gpu
        train_loss_giou_meter.avg: float, average giou loss on current process/gpu
        master_loss_ce_meter.avg: float, average ce loss on all processes/gpus
        master_loss_bbox_meter.avg: float, average bbox loss on all processes/gpus
        master_loss_giou_meter.avg: float, average giou loss on all processes/gpus
        train_time: float, training time
    """

    model.train()
    criterion.train()

    train_loss_ce_meter = AverageMeter()
    train_loss_bbox_meter = AverageMeter()
    train_loss_giou_meter = AverageMeter()
    master_loss_ce_meter = AverageMeter()
    master_loss_bbox_meter = AverageMeter()
    master_loss_giou_meter = AverageMeter()
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        samples = data[0]
        targets = data[1]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses.backward()
        if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
            optimizer.step()
            optimizer.clear_grad()

        # sync form other gpus for overall loss
        with paddle.no_grad():
            batch_size = paddle.to_tensor(samples.tensors.shape[0])
            master_loss_ce = loss_dict['loss_ce']
            master_loss_bbox = loss_dict['loss_bbox']
            master_loss_giou = loss_dict['loss_giou']
            master_batch_size = batch_size
            dist.all_reduce(master_loss_ce)
            dist.all_reduce(master_loss_bbox)
            dist.all_reduce(master_loss_giou)
            dist.all_reduce(master_batch_size)
            master_loss_ce = master_loss_ce / dist.get_world_size()
            master_loss_bbox = master_loss_bbox / dist.get_world_size()
            master_loss_giou = master_loss_giou / dist.get_world_size()
            master_loss_ce_meter.update(master_loss_ce.numpy()[0], master_batch_size.numpy()[0])
            master_loss_bbox_meter.update(master_loss_bbox.numpy()[0], master_batch_size.numpy()[0])
            master_loss_giou_meter.update(master_loss_giou.numpy()[0], master_batch_size.numpy()[0])
    
            train_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size.numpy()[0])
            train_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size.numpy()[0])
            train_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size.numpy()[0])
    
        if batch_id % debug_steps == 0:
            if local_logger:
                local_logger.info(
                    f"Epoch[{epoch:03d}/{total_epochs:03d}], " + 
                    f"Step[{batch_id:04d}/{total_batch:04d}], " + 
                    f"Avg loss_ce: {train_loss_ce_meter.avg:.4f}, " + 
                    f"Avg loss_bbox: {train_loss_bbox_meter.avg:.4f}, " + 
                    f"Avg loss_giou: {train_loss_giou_meter.avg:.4f}")
            if master_logger and dist.get_rank() == 0:
                master_logger.info(
                    f"Epoch[{epoch:03d}/{total_epochs:03d}], " + 
                    f"Step[{batch_id:04d}/{total_batch:04d}], " + 
                    f"Avg loss_ce: {master_loss_ce_meter.avg:.4f}, " + 
                    f"Avg loss_bbox: {master_loss_bbox_meter.avg:.4f}, " + 
                    f"Avg loss_giou: {master_loss_giou_meter.avg:.4f}") 

        dist.barrier()

    train_time = time.time() - time_st
    return (train_loss_ce_meter.avg,
            train_loss_bbox_meter.avg,
            train_loss_giou_meter.avg,
            master_loss_ce_meter.avg,
            master_loss_bbox_meter.avg,
            master_loss_giou_meter.avg,
            train_time)


def validate(dataloader,
             model,
             criterion,
             postprocessors,
             base_ds,
             total_batch,
             debug_steps=100,
             local_logger=None,
             master_logger=None):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: criterion
        postprocessors: postprocessor for generating bboxes
        base_ds: coco api for generate CocoEvaluator, pycocotools.coco.COCO(anno_file)
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_ce_meter.avg: float, average ce loss on current process/gpu
        val_loss_bbox_meter.avg: float, average bbox loss on current process/gpu
        val_loss_giou_meter.avg: float, average giou loss on current process/gpu
        master_loss_ce_meter.avg: float, average ce loss on all processes/gpus
        master_loss_bbox_meter.avg: float, average bbox loss on all processes/gpus
        master_loss_giou_meter.avg: float, average giou loss on all processes/gpus
        val_time: float, training time
    """
    model.eval()
    criterion.eval()

    val_loss_ce_meter = AverageMeter()
    val_loss_bbox_meter = AverageMeter()
    val_loss_giou_meter = AverageMeter()
    master_loss_ce_meter = AverageMeter()
    master_loss_bbox_meter = AverageMeter()
    master_loss_giou_meter = AverageMeter()

    time_st = time.time()

    iou_types = ('bbox', )
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            samples = data[0]
            targets = data[1]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # sync form other gpus for overall loss
            batch_size = paddle.to_tensor(samples.tensors.shape[0])
            master_loss_ce = loss_dict['loss_ce']
            master_loss_bbox = loss_dict['loss_bbox']
            master_loss_giou = loss_dict['loss_giou']
            master_batch_size = batch_size
            dist.all_reduce(master_loss_ce)
            dist.all_reduce(master_loss_bbox)
            dist.all_reduce(master_loss_giou)
            dist.all_reduce(master_batch_size)
            master_loss_ce = master_loss_ce / dist.get_world_size()
            master_loss_bbox = master_loss_bbox / dist.get_world_size()
            master_loss_giou = master_loss_giou / dist.get_world_size()
            master_loss_ce_meter.update(master_loss_ce.numpy()[0], master_batch_size.numpy()[0])
            master_loss_bbox_meter.update(master_loss_bbox.numpy()[0], master_batch_size.numpy()[0])
            master_loss_giou_meter.update(master_loss_giou.numpy()[0], master_batch_size.numpy()[0])

            val_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size.numpy()[0])
            val_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size.numpy()[0])
            val_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size.numpy()[0])
    
            if batch_id % debug_steps == 0:
                if local_logger:
                    local_logger.info(
                        f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                        f"Avg loss_ce: {val_loss_ce_meter.avg:.4f}, " +
                        f"Avg loss_bbox: {val_loss_bbox_meter.avg:.4f}, " +
                        f"Avg loss_giou: {val_loss_giou_meter.avg:.4f}, ")
                if master_logger and dist.get_rank() == 0:
                    master_logger.info(
                        f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                        f"Avg loss_ce: {master_loss_ce_meter.avg:.4f}, " +
                        f"Avg loss_bbox: {master_loss_bbox_meter.avg:.4f}, " +
                        f"Avg loss_giou: {master_loss_giou_meter.avg:.4f}, ")

            # coco evaluate
            orig_target_sizes = paddle.stack([t['orig_size'] for t in targets], axis=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id']: output for target, output in zip(targets, results)}
            
            if coco_evaluator is not None:
                coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    val_time = time.time() - time_st
    return (val_loss_ce_meter.avg,
            val_loss_bbox_meter.avg,
            val_loss_giou_meter.avg,
            master_loss_ce_meter.avg,
            master_loss_bbox_meter.avg,
            master_loss_giou_meter.avg,
            val_time)


def main_worker(*args):
    # STEP 0: Preparation
    config = args[0]
    dist.init_parallel_env()
    last_epoch = config.TRAIN.LAST_EPOCH
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
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
    if master_logger is not None:
        master_logger.info(f'----- world_size = {world_size}, local_rank = {local_rank}')
        
    # STEP 1: Create model
    model, criterion, postprocessors = build_detr(config)
    model = paddle.DataParallel(model)

    # STEP 2: Create train and val dataloader
    dataset_train, dataset_val = args[1], args[2]
    # create training dataloader
    total_batch_train = 0
    if not config.EVAL:
        dataloader_train = get_dataloader(dataset_train,
                                          batch_size=config.DATA.BATCH_SIZE,
                                          mode='train',
                                          multi_gpu=True)
        total_batch_train = len(dataloader_train)
        local_logger.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
        if master_logger is not None:
            master_logger.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
    # create validation dataloader
    dataloader_val = get_dataloader(dataset_val,
                                    batch_size=config.DATA.BATCH_SIZE_EVAL,
                                    mode='val',
                                    multi_gpu=True)
    total_batch_val = len(dataloader_val)
    local_logger.info(f'----- Total # of val batch (single gpu): {total_batch_val}')
    if master_logger is not None:
        master_logger.info(f'----- Total # of val batch (single gpu): {total_batch_val}')
    # create coco instance for validation
    base_ds = dataset_val.coco # pycocotools.coco.COCO(anno_file)

    # STEP 3: Define lr_scheduler
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
    elif config.TRAIN.LR_SCHEDULER.NAME == "multi-step":
        milestones = [int(v.strip()) for v in config.TRAIN.LR_SCHEDULER.MILESTONES.split(",")]
        scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.TRAIN.BASE_LR,
                                                       milestones=milestones,
                                                       gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                       last_epoch=last_epoch)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=config.TRAIN.BASE_LR,
                                                  step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
                                                  gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                  last_epoch=last_epoch)
    else:
        local_logger.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        if master_logger is not None:
            master_logger.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

    # STEP 4: Define optimizer
    params_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.stop_gradient is False]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.stop_gradient is False],
         "lr": config.MODEL.BACKBONE_LR}, # lr is lr_mult 
    ]

    if config.TRAIN.GRAD_CLIP:
        clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    else:
        clip = None
    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=model.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip,
            #apply_decay_param_fun=get_exclude_from_weight_decay_fn(['pos_embed', 'cls_token']),
            )
    else:
        local_logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        if master_logger is not None:
            master_logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # STEP 5: Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED + '.pdparams')
        model.set_dict(model_state)
        local_logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")
        if master_logger is not None:
            master_logger.info(
                f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME + '.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME + '.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME + '.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME + '.pdopt')
        optimizer.set_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")
        if master_logger is not None:
            master_logger.info(
                f"----- Resume Training: Load model state from {config.MODEL.RESUME}")
    
    # STEP 6: Validation
    if config.EVAL:
        local_logger.info('----- Start Validating')
        if master_logger is not None:
            master_logger.info('----- Start Validating')
        val_result = validate(dataloader=dataloader_val,
                              model=model,
                              criterion=criterion,
                              postprocessors=postprocessors,
                              base_ds=base_ds,
                              total_batch=total_batch_val,
                              debug_steps=config.REPORT_FREQ,
                              local_logger=local_logger,
                              master_logger=master_logger)
        val_loss_ce, val_loss_bbox, val_loss_giou = val_result[0], val_result[1], val_result[2]
        avg_loss_ce, avg_loss_bbox, avg_loss_giou = val_result[3], val_result[4], val_result[5]
        val_time = val_result[6]
        local_logger.info(f"Validation Loss_ce: {val_loss_ce:.4f}, " +
                          f"Validation Loss_bbox: {val_loss_bbox:.4f}, " +
                          f"Validation Loss_giou: {val_loss_giou:.4f}, " +
                          f"time: {val_time:.2f}")
        if master_logger is not None:
            master_logger.info(f"Validation Loss_ce: {avg_loss_ce:.4f}, " +
                               f"Validation Loss_bbox: {avg_loss_bbox:.4f}, " +
                               f"Validation Loss_giou: {avg_loss_giou:.4f}, " +
                               f"time: {val_time:.2f}")
        return

    # STEP 7: Start training and validation
    local_logger.info(f"Start training from epoch {last_epoch+1}.")
    if master_logger is not None:
        master_logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        local_logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        if master_logger is not None:
            master_logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        trn_res = train(dataloader=dataloader_train,
                        model=model,
                        criterion=criterion,
                        postprocessors=postprocessors,
                        base_ds=base_ds,
                        optimizer=optimizer,
                        epoch=epoch,
                        total_epochs=config.TRAIN.NUM_EPOCHS,
                        total_batch=total_batch_train,
                        debug_steps=config.REPORT_FREQ,
                        accum_iter=config.TRAIN.ACCUM_ITER,
                        local_logger=local_logger,
                        master_logger=master_logger)
        scheduler.step()

        trn_loss_ce, trn_loss_bbox, trn_loss_giou = trn_res[0], trn_res[1], trn_res[2]
        avg_loss_ce, avg_loss_bbox, avg_loss_giou = trn_res[3], trn_res[4], trn_res[5]
        trn_time = trn_res[6]

        local_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                          f"Loss_ce: {trn_loss_ce:.4f}, " +
                          f"Loss_bbox: {trn_loss_bbox:.4f}, " +
                          f"Loss_giou: {trn_loss_giou:.4f}, " +
                          f"time: {trn_time:.2f}")
        if master_logger is not None:
            master_logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                               f"Loss_ce: {avg_loss_ce:.4f}, " +
                               f"Loss_bbox: {avg_loss_bbox:.4f}, " +
                               f"Loss_giou: {avg_loss_giou:.4f}, " +
                               f"time: {trn_time:.2f}")

        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            local_logger.info('----- Start Validating')
            if master_logger is not None:
                master_logger.info('----- Start Validating')
            val_result = validate(dataloader=dataloader_val,
                                  model=model,
                                  criterion=criterion,
                                  postprocessors=postprocessors,
                                  base_ds=base_ds,
                                  total_batch=total_batch_val,
                                  debug_steps=config.REPORT_FREQ,
                                  local_logger=local_logger,
                                  master_logger=master_logger)
            val_loss_ce, val_loss_bbox, val_loss_giou = val_result[0], val_result[1], val_result[2]
            avg_loss_ce, avg_loss_bbox, avg_loss_giou = val_result[3], val_result[4], val_result[5]
            val_time = val_result[6]
            local_logger.info(f"Validation Loss_ce: {val_loss_ce:.4f}, " +
                              f"Validation Loss_bbox: {val_loss_bbox:.4f}, " +
                              f"Validation Loss_giou: {val_loss_giou:.4f}, " +
                              f"time: {val_time:.2f}")
            if master_logger is not None:
                master_logger.info(f"Validation Loss_ce: {avg_loss_ce:.4f}, " +
                                   f"Validation Loss_bbox: {avg_loss_bbox:.4f}, " +
                                   f"Validation Loss_giou: {avg_loss_giou:.4f}, " +
                                   f"time: {val_time:.2f}")
        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss_ce}")
                paddle.save(model.state_dict(), model_path + '.pdparams')
                paddle.save(optimizer.state_dict(), model_path + '.pdopt')
                logger.info(f"----- Save model: {model_path}.pdparams")
                logger.info(f"----- Save optim: {model_path}.pdopt")


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
        dataset_train = build_coco('train', config.DATA.DATA_PATH)
    else:
        dataset_train = None
    dataset_val = build_coco('val', config.DATA.DATA_PATH)
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(config, dataset_train, dataset_val, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
