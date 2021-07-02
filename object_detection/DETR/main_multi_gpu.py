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
from detr import build_detr
from utils import AverageMeter
from utils import WarmupCosineScheduler
from config import get_config
from config import update_config


parser = argparse.ArgumentParser('DETR')
parser.add_argument('-cfg', type=str, default=None)
parser.add_argument('-dataset', type=str, default=None)
parser.add_argument('-batch_size', type=int, default=None)
parser.add_argument('-data_path', type=str, default=None)
parser.add_argument('-backbone', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-pretrained', type=str, default=None)
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

#config.freeze()

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

# set logging format
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')


def train(dataloader,
          model,
          criterion,
          postprocessors,
          base_ds,
          optimizer,
          epoch,
          total_batch,
          debug_steps=100,
          accum_iter=1):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, DETR model
        criterion: nn.Layer
        postprocessors: nn.Layer
        base_ds: coco api instance
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_ce_meter.avg
        train_loss_bbox_meter.avg
        train_loss_giou_meter.avg
        train_time
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
        #targets = [{k:v for k,v in t.items()} for t in targets]
            
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        losses.backward()
        if ((batch_id +1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
            optimizer.step()
            optimizer.clear_grad()

        # logging losses
        batch_size = samples.tensors.shape[0]
        train_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size)
        train_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size)
        train_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size)
    
        if batch_id > 0 and batch_id % debug_steps == 0:
            logger.info(
                f"Train Step[{batch_id:04d}/{total_batch:04d}], " + 
                f"Avg loss_ce: {train_loss_ce_meter.avg:.4f}, " + 
                f"Avg loss_bbox: {train_loss_bbox_meter.avg:.4f}, " + 
                f"Avg loss_giou: {train_loss_giou_meter.avg:.4f}, ") 

    train_time = time.time() - time_st
    return train_loss_ce_meter.avg, train_loss_bbox_meter.avg, train_loss_giou_meter.avg, train_time


def validate(dataloader, model, criterion, postprocessors, base_ds, total_batch, debug_steps=100):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: criterion
        postprocessors: postprocessor for generating bboxes
        base_ds: COCO instance
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
    Returns:
        val_loss_meter.avg
        val_acc_meter.avg
        val_time
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
            targets = [{k:v for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # logging val losses
            dist.all_reduce(loss_dict['loss_ce'])
            dist.all_reduce(loss_dict['loss_bbox'])
            dist.all_reduce(loss_dict['loss_giou'])
            loss_dict['loss_ce'] /= dist.get_world_size()
            loss_dict['loss_bbox'] /= dist.get_world_size()
            loss_dict['loss_giou'] /= dist.get_world_size()

            batch_size = paddle.to_tensor(samples.tensors.shape[0])
            dist.all_reduce(batch_size)
            batch_size = batch_size.numpy()[0]
            val_loss_ce_meter.update(loss_dict['loss_ce'].numpy()[0], batch_size)
            val_loss_bbox_meter.update(loss_dict['loss_bbox'].numpy()[0], batch_size)
            val_loss_giou_meter.update(loss_dict['loss_giou'].numpy()[0], batch_size)

            if batch_id > 0 and batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Avg loss_ce: {val_loss_ce_meter.avg:.4f}, " +
                    f"Avg loss_bbox: {val_loss_bbox_meter.avg:.4f}, " +
                    f"Avg loss_giou: {val_loss_giou_meter.avg:.4f}, ")

            # coco evaluate
            orig_target_sizes = paddle.stack([t['orig_size'] for t in targets], axis=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            res = {target['image_id'].cpu().numpy()[0]: output for target, output in zip(targets, results)}
            
            if coco_evaluator is not None:
                coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    val_time = time.time() - time_st
    return val_loss_ce_meter.avg, val_loss_bbox_meter.avg, val_loss_giou_meter.avg, val_time


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
    model, criterion, postprocessors = build_detr(config)
    model = paddle.DataParallel(model)
    # 2. Create train and val dataloader
    dataset_train, dataset_val = args[0], args[1]
    total_batch_train = 0
    if not config.EVAL:
        dataloader_train = get_dataloader(dataset_train,
                                      batch_size=config.DATA.BATCH_SIZE,
                                      mode='train',
                                      multi_gpu=True)
        total_batch_train = len(dataloader_train)

    dataloader_val = get_dataloader(dataset_val,
                                batch_size=config.DATA.BATCH_SIZE_EVAL,
                                mode='val',
                                multi_gpu=True)
    total_batch_val = len(dataloader_val)
    base_ds = dataset_val.coco # pycocotools.coco.COCO(anno_file)

    logging.info(f'----- Total # of train batch (single gpu): {total_batch_train}')
    logging.info(f'----- Total # of val batch (single gpu): {total_batch_val}')
    # 4. Define optimizer and lr_scheduler
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
        logging.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
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
            #apply_decay_param_fun=get_exclude_from_weight_decay_fn(['pos_embed', 'cls_token']),
            )
    else:
        logging.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # 5. Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME+'.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME+'.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")
    
    # 6. Validation
    if config.EVAL:
        logger.info('----- Start Validating')
        val_loss_ce, val_loss_bbox, val_loss_giou, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            base_ds=base_ds,
            total_batch=total_batch_val,
            debug_steps=config.REPORT_FREQ)
        logger.info(f"Validation Loss_ce: {val_loss_ce:.4f}, " +
                    f"Validation Loss_bbox: {val_loss_bbox:.4f}, " +
                    f"Validation Loss_giou: {val_loss_giou:.4f}, " +
                    f"time: {val_time:.2f}")
        return

    # 6. Start training and validation
    logging.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logging.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_acc, train_time = train(dataloader=dataloader_train,
                                                  model=model,
                                                  criterion=criterion,
                                                  optimizer=optimizer,
                                                  epoch=epoch,
                                                  total_batch=total_batch_train,
                                                  debug_steps=config.REPORT_FREQ,
                                                  accum_iter=config.TRAIN.ACCUM_ITER)
        scheduler.step()

        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
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
                total_batch=total_batch_val,
                debug_steps=config.REPORT_FREQ)
            logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                        f"Validation Loss_ce: {val_loss_ce:.4f}, " +
                        f"Validation Loss_bbox: {val_loss_bbox:.4f}, " +
                        f"Validation Loss_giou: {val_loss_giou:.4f}, " +
                        f"time: {val_time:.2f}")
        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path)
                paddle.save(optimizer.state_dict(), model_path)
                logger.info(f"----- Save model: {model_path}.pdparams")
                logger.info(f"----- Save optim: {model_path}.pdopt")


def main():
    if not config.EVAL:
        dataset_train = build_coco('train', config.DATA.DATA_PATH)
    else:
        dataset_train = None
    dataset_val = build_coco('val', config.DATA.DATA_PATH)
    config.NGPUS = len(paddle.static.cuda_places()) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(dataset_train, dataset_val, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
