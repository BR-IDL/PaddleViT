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

"""ViT training/validation using multiple GPU """

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
from config import get_config
from config import update_config


parser = argparse.ArgumentParser('ViT')
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
parser.add_argument('-amp', action='store_true')
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
    config.SAVE = '{}/train-{}'.format(config.SAVE,
                                       time.strftime('%Y%m%d-%H-%M-%S'))
else:
    config.SAVE = '{}/eval-{}'.format(config.SAVE,
                                      time.strftime('%Y%m%d-%H-%M-%S'))

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
          optimizer,
          epoch,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          amp=False):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        amp: bool, if True, use mix precision training, default: False
    Returns:
        train_loss_meter.avg
        train_time
    """
    model.train()
    train_loss_meter = AverageMeter()
    if amp is True:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        image = data[0]

        if amp is True:
            with paddle.amp.auto_cast():
                output, target = model(image)
                loss = criterion(output, target)
            scaled = scaler.scale(loss)
            scaled.backward()

            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()

        else:
            output, target = model(image)
            loss = criterion(output, target)
            # NOTE: division may be needed depending on the loss function
            # Here no division is needed:
            # default 'reduction' param in nn.CrossEntropyLoss is set to 'mean'
            # loss =  loss / accum_iter
            loss.backward()

            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()

        batch_size = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}")

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time


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
    model = build_model(config)
    model = paddle.DataParallel(model)
    # 2. Create train dataloader
    dataset_train = args[0]
    dataloader_train = get_dataloader(config, dataset_train, 'train', True)
    total_batch_train = len(dataloader_train)
    logging.info(
        f'----- Total # of train batch (single gpu): {total_batch_train}')
    # 3. Define criterion
    criterion = nn.MSELoss()
    # 4. Define lr_scheduler
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
        milestones = [int(v.strip())
                      for v in config.TRAIN.LR_SCHEDULER.MILESTONES.split(",")]
        scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.TRAIN.BASE_LR,
                                                       milestones=milestones,
                                                       gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
                                                       last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(
            f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
    # 5. Define optimizer
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
        raise NotImplementedError(
            f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
    # 6. Load pretrained model or load resume model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(
                f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        logger.info(
            f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME+'.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME+'.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")

    # 7. Start training and validation
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # train
        logging.info(
            f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_time = train(dataloader=dataloader_train,
                                       model=model,
                                       criterion=criterion,
                                       optimizer=optimizer,
                                       epoch=epoch,
                                       total_batch=total_batch_train,
                                       debug_steps=config.REPORT_FREQ,
                                       accum_iter=config.TRAIN.ACCUM_ITER,
                                       amp=config.AMP)
        scheduler.step()

        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        # No need to do validation during pretraining

        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path + '.pdparams')
                paddle.save(optimizer.state_dict(), model_path + '.pdopt')
                logger.info(f"----- Save model: {model_path}.pdparams")
                logger.info(f"----- Save optim: {model_path}.pdopt")


def main():
    dataset_train = get_dataset(config, mode='train')
    config.NGPUS = len(paddle.static.cuda_places()
                       ) if config.NGPUS == -1 else config.NGPUS
    dist.spawn(main_worker, args=(dataset_train, ), nprocs=config.NGPUS)


if __name__ == "__main__":
    main()
