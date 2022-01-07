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

""" DETR training using single GPU, this is just a demo, please use multi gpu version"""
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
from coco import build_coco
from coco import get_dataloader
from coco_eval import CocoEvaluator
from config import get_config
from config import update_config
from utils import WarmupCosineScheduler
from utils import AverageMeter
from detr import build_detr


def get_arguments():
    """ return arguments, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('DETR')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default="coco")
    parser.add_argument('-batch_size', type=int, default=2)
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
          logger=None):
    """ Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, DETR model
        criterion: criterion defined in DETR
        postprocessors: PostProcess, converts output to the coco format
        base_ds: coco api instance for generate CocoEvaluator, pycocotools.coco.COCO(anno_file)
        optimizer: nn.optimizer
        epoch: int, current epoch
        total_epochs: int, total num of epochs
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        logger: logger for logging, default: None 
    Returns:
        train_loss_ce_meter.avg: float, average ce loss on current process/gpu
        train_loss_bbox_meter.avg: float, average bbox loss on current process/gpu
        train_loss_giou_meter.avg: float, average giou loss on current process/gpu
        train_time: float, training time
    """
    model.train()
    criterion.train()

    train_loss_ce_meter = AverageMeter()
    train_loss_bbox_meter = AverageMeter()
    train_loss_giou_meter = AverageMeter()

    time_st = time.time()

    iou_types = ('bbox', )
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for batch_id, data in enumerate(dataloader):
        samples = data[0]
        targets = data[1]
        if samples is None:
            print('skip None')
            continue
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses.backward()
        if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
            optimizer.step()
            optimizer.clear_grad()

        # logging losses
        batch_size = samples.tensors.shape[0]
        train_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size)
        train_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size)
        train_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size)
    
        if logger and batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{total_epochs:03d}], " + 
                f"Step[{batch_id:04d}/{total_batch:04d}], " + 
                f"Avg loss_ce: {train_loss_ce_meter.avg:.4f}, " + 
                f"Avg loss_bbox: {train_loss_bbox_meter.avg:.4f}, " + 
                f"Avg loss_giou: {train_loss_giou_meter.avg:.4f}") 

    train_time = time.time() - time_st
    return train_loss_ce_meter.avg, train_loss_bbox_meter.avg, train_loss_giou_meter.avg, train_time


def validate(dataloader,
             model,
             criterion,
             postprocessors,
             base_ds,
             total_batch,
             debug_steps=100,
             logger=None):
    """ Validate for whole dataset 
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, DETR model
        criterion: criterion defined in DETR
        postprocessors: PostProcess, converts output to the coco format
        base_ds: coco api instance for generate CocoEvaluator, pycocotools.coco.COCO(anno_file)
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None 
    Returns:
        val_loss_ce_meter.avg: float, average ce loss on current process/gpu
        val_loss_bbox_meter.avg: float, average bbox loss on current process/gpu
        val_loss_giou_meter.avg: float, average giou loss on current process/gpu
        val_time: float, validation time
    """
    model.eval()
    criterion.eval()

    val_loss_ce_meter = AverageMeter()
    val_loss_bbox_meter = AverageMeter()
    val_loss_giou_meter = AverageMeter()

    time_st = time.time()

    iou_types = ('bbox', )
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            samples = data[0]
            targets = data[1]
            if samples is None:
                continue
            
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # logging val losses
            batch_size = samples.tensors.shape[0]
            val_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size)
            val_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size)
            val_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size)
    
            if logger and batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " + 
                    f"Avg loss_ce: {val_loss_ce_meter.avg:.4f}, " + 
                    f"Avg loss_bbox: {val_loss_bbox_meter.avg:.4f}, " + 
                    f"Avg loss_giou: {val_loss_giou_meter.avg:.4f}") 

            # coco evaluate
            orig_target_sizes = paddle.stack([t['orig_size'] for t in targets], axis=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id']: output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize() #TODO: get stats[0] and return mAP

    val_time = time.time() - time_st
    return val_loss_ce_meter.avg, val_loss_bbox_meter.avg, val_loss_giou_meter.avg, val_time


def main():
    # STEP 0: Preparation
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

    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger = get_logger(filename=os.path.join(config.SAVE, 'log.txt'))
    logger.info(f'\n{config}')

    # STEP 1: Create model and criterion
    model, criterion, postprocessors = build_detr(config)
    # STEP 2: Create train and val dataloader
    if not config.EVAL:
        dataset_train = build_coco('train', config.DATA.DATA_PATH)
        dataloader_train = get_dataloader(dataset_train,
                                          batch_size=config.DATA.BATCH_SIZE,
                                          mode='train', 
                                          multi_gpu=False)

    dataset_val = build_coco('val', config.DATA.DATA_PATH)
    dataloader_val = get_dataloader(dataset_val,
                                    batch_size=config.DATA.BATCH_SIZE_EVAL,
                                    mode='val', 
                                    multi_gpu=False)

    base_ds = dataset_val.coco   # pycocotools.coco.COCO(anno_file)
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
                                                       milestones=milestons,
                                                       gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                       last_epoch=last_epoch)
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=config.TRAIN.BASE_LR,
                                                  step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
                                                  gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                  last_epoch=last_epoch)
    else:
        logger.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

    # STEP 4: Define optimizer
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
            )
    else:
        logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # STEP 5: Load pretrained model or load resume model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED + '.pdparams') 
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME: 
        assert os.path.isfile(config.MODEL.RESUME + '.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME + '.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME + '.pdparams') 
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME + '.pdopt') 
        optimizer.set_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")

    # STEP 6: Validation
    if config.EVAL:
        logger.info(f'----- Start Validating')
        val_loss_ce, val_loss_bbox, val_loss_giou, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            base_ds=base_ds,
            total_batch=len(dataloader_val),
            debug_steps=config.REPORT_FREQ)
        logger.info(f"Validation Loss ce: {val_loss_ce:.4f}, " +
                    f"Validation Loss bbox: {val_loss_bbox:.4f}, " +
                    f"Validation Loss giou: {val_loss_giou:.4f}, " +
                    f"time: {val_time:.2f}")
        return

    # STEP 7: Start training and validation
    logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss_ce, train_loss_bbox, train_loss_giou, train_time = train(
            dataloader=dataloader_train,
            model=model, 
            criterion=criterion, 
            postprocessors=postprocessors,
            base_ds=base_ds,
            optimizer=optimizer, 
            epoch=epoch,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batch=len(dataloader_train),
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            logger=logger)
        scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss ce: {train_loss_ce:.4f}, " +
                    f"Train Loss bbox: {train_loss_bbox:.4f}, " +
                    f"Train Loss giou: {train_loss_giou:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            logger.info(f'----- Validation after Epoch: {epoch}')
            val_loss_ce, val_loss_bbox, val_loss_giou, val_time = validate(
                dataloader=dataloader_val,
                model=model,
                criterion=criterion,
                postprocessors=postprocessors,
                base_ds=base_ds,
                total_batch=len(dataloader_val),
                debug_steps=config.REPORT_FREQ,
                logger=logger)
            logger.info(f"Validation Loss ce: {val_loss_ce:.4f}, " +
                        f"Validation Loss bbox: {val_loss_bbox:.4f}, " +
                        f"Validation Loss giou: {val_loss_giou:.4f}, " +
                        f"time: {val_time:.2f}")
        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(
                config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss_ce}")
            paddle.save(model.state_dict(), model_path + '.pdparams')
            paddle.save(optimizer.state_dict(), model_path + '.pdopt')
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")


if __name__ == "__main__":
    main()
