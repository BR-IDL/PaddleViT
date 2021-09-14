#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
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

import sys
import os
import time
import logging
import argparse
import random
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from coco import build_coco
from coco import get_dataloader
from coco_eval import CocoEvaluator
from pvtv2_det import build_pvtv2_det as build_det_model
from config import get_config
from config import update_config
from utils import WarmupCosineScheduler
from utils import AverageMeter


parser = argparse.ArgumentParser('PVTv2-Det')
parser.add_argument('-cfg', type=str, default='./configs/pvtv2_b0.yaml')
parser.add_argument('-dataset', type=str, default="coco")
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-data_path', type=str, default='/dataset/coco/')
parser.add_argument('-backbone', type=str, default=None)
parser.add_argument('-ngpus', type=int, default=None)
parser.add_argument('-pretrained', type=str, default=None)
parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-last_epoch', type=int, default=None)
parser.add_argument('-eval', action='store_true')
args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")

config = get_config()
config = update_config(config, args)

if not config.EVAL:
    config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
else:
    config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

config.freeze()

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')


def train(dataloader, model, base_ds, optimizer, epoch, total_batch, debug_steps=100, accum_iter=1):
    model.train()

    train_loss_cls_meter = AverageMeter()
    train_loss_reg_meter = AverageMeter()
    train_loss_rpn_cls_meter = AverageMeter()
    train_loss_rpn_reg_meter = AverageMeter()

    time_st = time.time()

    #iou_types = ('bbox', )
    #coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for batch_id, data in enumerate(dataloader):
        samples = data[0]
        targets = data[1]
            
        loss_dict = model(samples, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()

        if ((batch_id +1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
            optimizer.step()
            optimizer.clear_grad()

        # logging losses
        batch_size = samples.tensors.shape[0]
        train_loss_cls_meter.update(loss_dict['loss_cls'].numpy()[0], batch_size)
        train_loss_reg_meter.update(loss_dict['loss_reg'].numpy()[0], batch_size)
        train_loss_rpn_cls_meter.update(loss_dict['loss_rpn_cls'].numpy()[0], batch_size)
        train_loss_rpn_reg_meter.update(loss_dict['loss_rpn_reg'].numpy()[0], batch_size)
    
        if batch_id > 0 and batch_id % debug_steps == 0:
            logger.info(
                f"Train Step[{batch_id:04d}/{total_batch:04d}], " + 
                f"Avg loss_cls: {train_loss_cls_meter.avg:.4f}, " + 
                f"Avg loss_reg: {train_loss_reg_meter.avg:.4f}, " + 
                f"Avg loss_rpn_cls: {train_loss_rpn_cls_meter.avg:.4f}, " + 
                f"Avg loss_rpn_reg: {train_loss_rpn_reg_meter.avg:.4f}") 

    train_time = time.time() - time_st
    return (train_loss_cls_meter.avg,
            train_loss_reg_meter.avg,
            train_loss_rpn_cls_meter.avg,
            train_loss_rpn_reg_meter.avg,
            train_time)


def validate(dataloader, model, base_ds, total_batch, debug_steps=100):
    model.eval()
    time_st = time.time()

    iou_types = ('bbox', )
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            samples = data[0]
            targets = data[1]

            ## save temp data and targets for testing            
            #tar = {}
            #for key, val in targets.items():
            #    print(key)
            #    tar[key] = []
            #    for item in targets[key]:
            #        tar[key].append(item.cpu().numpy())
            #with open('t.npy', 'wb') as ofile:
            #    np.save(ofile, samples.tensors.cpu().numpy())
            #    np.save(ofile, samples.mask.cpu().numpy())
            #    np.save(ofile, tar)
            #
            #break

            prediction = model(samples, targets)

            if batch_id > 0 and batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}], done") 

            #res = {target_id: output for target_id, output in zip(targets['image_id'], prediction)}
            res = {}
            for target_id, output in zip(targets['image_id'], prediction):
                target_id = target_id.cpu().numpy()[0]
                output = output.cpu().numpy()
                if output.shape[0] != 0:
                    pred_dict = {'boxes': output[:, 2::],
                                 'scores': output[:, 1],
                                 'labels': output[:, 0]}
                    res[int(target_id)] = pred_dict
                else:
                    res[int(target_id)] = {}

            if coco_evaluator is not None:
                coco_evaluator.update(res)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize() #TODO: get stats[0] and return mAP

    val_time = time.time() - time_st
    return val_time


def main():
    # 0. Preparation
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 1. Create model and criterion
    model = build_det_model(config)
    # 2. Create train and val dataloader
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
    # 3. Define lr_scheduler
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
                                                       milestones=milestons,
                                                       gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                       last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

    # 5. Define optimizer
    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        if config.TRAIN.GRAD_CLIP:
            clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
        else:
            clip = None
        optimizer = paddle.optimizer.Momentum(parameters=model.parameters(),
                                     learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
                                     weight_decay=config.TRAIN.WEIGHT_DECAY,
                                     momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                     grad_clip=clip,
                                     )
    elif config.TRAIN.OPTIMIZER.NAME == "AdamW":
        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(),
                                       beta1=config.TRAIN.OPTIMIZER.BETAS[0],
                                       beta2=config.TRAIN.OPTIMIZER.BETAS[1],
                                       epsilon=config.TRAIN.OPTIMIZER.EPS,
                                       )
    else:
        logging.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # 6. Load pretrained model or load resume model and optimizer states
    if config.MODEL.PRETRAINED:
    #if config.MODEL.PRETRAINED and os.path.isfile(config.MODEL.PRETRAINED + '.pdparams'):
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams')
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams') 

        # if from classification weights, add prefix 'backbone' and set state dict
        if sum(['backbone' in key for key in model_state.keys()]) == 0:
            logger.info(f"----- Pretrained: Load backbone from {config.MODEL.PRETRAINED}")
            new_model_state = dict()
            for key, val in model_state.items():
                new_model_state['backbone.' + key] = val
            model.set_state_dict(new_model_state)
        else:
            logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")
            model.set_state_dict(model_state)

    if config.MODEL.RESUME and os.path.isfile(config.MODEL.RESUME+'.pdparams') and os.path.isfile(config.MODEL.RESUME+'.pdopt'):
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams') 
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt') 
        optimizer.set_dict(opt_state)
        logger.info(f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")

    # 7. Validation
    if config.EVAL:
        logger.info(f'----- Start Validating')
        val_time, all_eval_result = validate(
            dataloader=dataloader_val,
            model=model,
            base_ds=base_ds,
            total_batch=len(dataloader_val),
            debug_steps=config.REPORT_FREQ)
 
        logger.info('IoU metric: bbox')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[0]:0.3f}')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[1]:0.3f}')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.75":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[2]:0.3f}')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={" small":>6s} | maxDets={100:>3d} ] = {all_eval_result[3]:0.3f}')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={"medium":>6s} | maxDets={100:>3d} ] = {all_eval_result[4]:0.3f}')
        logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={" large":>6s} | maxDets={100:>3d} ] = {all_eval_result[5]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={1:>3d} ] = {all_eval_result[6]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={10:>3d} ] = {all_eval_result[7]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[8]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"small":>6s} | maxDets={100:>3d} ] = {all_eval_result[9]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"medium":>6s} | maxDets={100:>3d} ] = {all_eval_result[10]:0.3f}')
        logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"large":>6s} | maxDets={100:>3d} ] = {all_eval_result[11]:0.3f}')
        logger.info(f"Val time: {val_time:.2f}")
        return

    # 8. Start training and validation
    logging.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logging.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss_cls, train_loss_reg, train_loss_rpn_cls, train_loss_rpn_reg, train_time = train(
            dataloader=dataloader_train,
            model=model, 
            base_ds=base_ds,
            optimizer=optimizer, 
            epoch=epoch,
            total_batch=len(dataloader_train),
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER)
        scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss cls: {train_loss_cls:.4f}, " +
                    f"Train Loss reg: {train_loss_reg:.4f}, " +
                    f"Train Loss rpn cls: {train_loss_rpn_cls:.4f}, " +
                    f"Train Loss rpn reg: {train_loss_rpn_reg:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            logger.info(f'----- Validation after Epoch: {epoch}')
            val_time, all_eval_result = validate(
        	    dataloader=dataloader_val,
        	    model=model,
        	    base_ds=base_ds,
        	    total_batch=len(dataloader_val),
        	    debug_steps=config.REPORT_FREQ)
 
            logger.info('IoU metric: bbox')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[0]:0.3f}')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[1]:0.3f}')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.75":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[2]:0.3f}')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={" small":>6s} | maxDets={100:>3d} ] = {all_eval_result[3]:0.3f}')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={"medium":>6s} | maxDets={100:>3d} ] = {all_eval_result[4]:0.3f}')
            logger.info(f'{"Average Precision":<18} (AP) @[ IoU={"0.50:0.95":<9} | area={" large":>6s} | maxDets={100:>3d} ] = {all_eval_result[5]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={1:>3d} ] = {all_eval_result[6]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={10:>3d} ] = {all_eval_result[7]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"all":>6s} | maxDets={100:>3d} ] = {all_eval_result[8]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"small":>6s} | maxDets={100:>3d} ] = {all_eval_result[9]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"medium":>6s} | maxDets={100:>3d} ] = {all_eval_result[10]:0.3f}')
            logger.info(f'{"Average Recall":<18} (AR) @[ IoU={"0.50:0.95":<9} | area={"large":>6s} | maxDets={100:>3d} ] = {all_eval_result[11]:0.3f}')
            logger.info(f"Val time: {val_time:.2f}")

        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
            paddle.save(model.state_dict(), model_path + '.pdparams')
            paddle.save(optimizer.state_dict(), model_path + '.pdopt')
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")


if __name__ == "__main__":
    main()
