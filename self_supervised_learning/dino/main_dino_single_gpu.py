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

"""ViT DINO training/validation using single GPU """

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
from datasets import get_dataloader
from datasets import get_dataset
import utils
from utils import AverageMeter
from utils import WarmupCosineScheduler
from utils import get_params_groups
from config import get_config
from config import update_config
from transformer import MultiCropWrapper
from transformer import DINOHead
from transformer import build_vit as build_model


def get_arguments():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('ViT Dino')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-num_classes', type=int, default=None)
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
            master_logger.info(msg_master)
        elif level == 'fatal':
            master_logger.fatal(msg_master)
        else:
            raise ValueError("level must in ['info', 'fatal']")


def train(dataloader,
          student_model,
          teacher_model,
          criterion,
          optimizer,
          lr_schedule,
          wd_schedule,
          momentum_schedule,
          freeze_last_layer,
          epoch,
          total_epochs,
          total_batch,
          debug_steps=100,
          accum_iter=1,
          amp=False,
          logger=None):
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
        amp: bool, if True, use mix precision training, default: False
        logger: logger for logging, default: None
    Returns:
        train_loss_meter.avg: float, average loss on current process/gpu
        train_acc_meter.avg: float, average top1 accuracy on current process/gpu
        train_time: float, training time
    """
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()

    if amp is True:
        scaler = paddle.amp.GradScaler() # default init_loss_scaling=32768
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        images = data[0]
        #label = data[1]

        # set lr using scheduler
        global_train_iter = len(dataloader) * (epoch-1) + batch_id # epoch starts from 1
        optimizer.set_lr(lr_schedule[global_train_iter])
        # set wd using scheduler
        optimizer.regularization = paddle.regularizer.L2Decay(wd_schedule[global_train_iter])

        with paddle.amp.auto_cast(amp is True):
            teacher_output = teacher_model(images[:2]) # only the 2 global views pass the teacher
            student_output = student_model(images)
            loss = criterion(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            message = f'Loss is {loss.item()}, stopping training'
            write_log(local_logger, None, message, 'fatal')
            sys.exit(1)

        # student update
        optimizer.clear_grad()

        if not amp: #fp32
            loss.backward()
            utils.cancle_gradient_last_layer(epoch, student_model, freeze_last_layer)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            utils.cancle_gradient_last_layer(epoch, student_model, freeze_last_layer)
            scaler.step(optimizer)
            scaler.update()

        # EMA update for the teacher
        with paddle.no_grad():
            m = momentum_schedule[global_train_iter] # momemtum parameter
            for param_q, param_k in zip(student_model.parameters(), teacher_model.parameters()):
                new_w = (param_k * m) + (1 - m) * param_q.detach()
                param_k.set_value(new_w)

        batch_size = images[0].shape[0]
        train_loss_meter.update(loss.numpy()[0], batch_size)

        if logger and batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}")

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time


#def validate(dataloader, model, criterion, total_batch, debug_steps=100, logger=None):
#    """Validation for whole dataset
#    Args:
#        dataloader: paddle.io.DataLoader, dataloader instance
#        model: nn.Layer, a ViT model
#        criterion: nn.criterion
#        total_batch: int, total num of batches for one epoch
#        debug_steps: int, num of iters to log info, default: 100
#        logger: logger for logging, default: None
#    Returns:
#        val_loss_meter.avg: float, average loss on current process/gpu
#        val_acc1_meter.avg: float, average top1 accuracy on current process/gpu
#        val_acc5_meter.avg: float, average top5 accuracy on current process/gpu
#        val_time: float, valitaion time
#    """
#    model.eval()
#    val_loss_meter = AverageMeter()
#    val_acc1_meter = AverageMeter()
#    val_acc5_meter = AverageMeter()
#    time_st = time.time()
#
#    with paddle.no_grad():
#        for batch_id, data in enumerate(dataloader):
#            image = data[0]
#            label = data[1]
#
#            output = model(image)
#            loss = criterion(output, label)
#
#            pred = F.softmax(output)
#            acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1))
#            acc5 = paddle.metric.accuracy(pred, label.unsqueeze(1), k=5)
#
#            batch_size = image.shape[0]
#            val_loss_meter.update(loss.numpy()[0], batch_size)
#            val_acc1_meter.update(acc1.numpy()[0], batch_size)
#            val_acc5_meter.update(acc5.numpy()[0], batch_size)
#
#            if logger and batch_id % debug_steps == 0:
#                logger.info(
#                    f"Val Step[{batch_id:04d}/{total_batch:04d}], " +
#                    f"Avg Loss: {val_loss_meter.avg:.4f}, " +
#                    f"Avg Acc@1: {val_acc1_meter.avg:.4f}, " +
#                    f"Avg Acc@5: {val_acc5_meter.avg:.4f}")
#
#    val_time = time.time() - time_st
#    return val_loss_meter.avg, val_acc1_meter.avg, val_acc5_meter.avg, val_time


class DINOLoss(nn.Layer):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
        warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer('center', paddle.zeros([1, out_dim]))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue # skip case teacher and student operate on the same view
                loss = paddle.sum(-q * F.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @paddle.no_grad()
    def update_center(self, teacher_output):
        batch_center = paddle.sum(teacher_output, axis=0, keepdim=True)
        #dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size()) 
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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

    # STEP 1: Create student and teacher model
    student_model = build_model(config) # student use droppath
    config.MODEL.DROPPATH = 0.0
    teacher_model = build_model(config) # teacher no droppath

    # multi-crop wrapper handles forward with inputs of different resolutions
    student_model = MultiCropWrapper(
        student_model,
        DINOHead(in_dim=config.MODEL.TRANS.EMBED_DIM,
                 out_dim=config.MODEL.OUT_DIM, # 65536 works well
                 use_bn=False,
                 norm_last_layer=True),
    )
    teacher_model = MultiCropWrapper(
        teacher_model,
        DINOHead(in_dim=config.MODEL.TRANS.EMBED_DIM,
                 out_dim=config.MODEL.OUT_DIM,
                 use_bn=False),
    )

    ## TODO: sync batch norms
    #if utils.has_batchnorms(student_model):
    #    student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    #    teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    #    teacher_model_without_ddp = copy.deepcopy(teacher_model)
    #    teacher_model = paddle.DataParallel(teacher_model)
    #    # TODO: check   
    #else:
    #    # TODO: check   
    #    teacher_model_without_ddp = teacher_model
    #student_model = paddle.DataParallel(student_model)

    # teacher and student start with the same weights
    teacher_model.set_state_dict(student_model.state_dict())
    # no bp throught teacher
    for p in teacher_model.parameters():
        p.stop_gradient = True

    # STEP 2: Create train dataloader
    dataset_train = get_dataset(config, mode='train')
    dataloader_train = get_dataloader(config, dataset_train, 'train', False)

    # 3. Define criterion
    #criterion = nn.CrossEntropyLoss()
    criterion = DINOLoss(config.MODEL.OUT_DIM,
                         config.DATA.LOCAL_CROPS_NUMBER + 2,
                         config.TRAIN.WARMUP_TEACHER_TEMP,
                         config.TRAIN.TEACHER_TEMP,
                         config.TRAIN.WARMUP_TEACHER_TEMP_EPOCHS,
                         config.TRAIN.NUM_EPOCHS)

    # 4. Define lr_scheduler
    lr_schedule = utils.cosine_scheduler(config.TRAIN.BASE_LR, # add linear scale
                                         config.TRAIN.END_LR,
                                         config.TRAIN.NUM_EPOCHS,
                                         len(dataloader_train),
                                         warmup_epochs=config.TRAIN.WARMUP_EPOCHS)
    wd_schedule = utils.cosine_scheduler(config.TRAIN.WEIGHT_DECAY,
                                         config.TRAIN.WEIGHT_DECAY_END,
                                         config.TRAIN.NUM_EPOCHS,
                                         len(dataloader_train))
    momentum_schedule = utils.cosine_scheduler(config.TRAIN.MOMENTUM_TEACHER,
                                               1,
                                               config.TRAIN.NUM_EPOCHS,
                                               len(dataloader_train))
    
    params_groups = utils.get_params_groups(student_model)

    if config.TRAIN.GRAD_CLIP:
        clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    else:
        clip = None

    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=params_gropus,
            learning_rate=0., # set by scheduler
            weight_decay=0., # set by scheduler
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=params_groups,
            learning_rate=0., # set by scheduler
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=0., # set by scheduler
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip)
    else:
        logger.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # STEP 6: Load pretrained model or load resume model and optimizer states
    if config.MODEL.PRETRAINED:
        if (config.MODEL.PRETRAINED).endswith('.pdparams'):
            raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")


    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME + '.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME + '_dino_loss.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME + '.pdopt') is True
        model_state = paddle.load(config.MODEL.RESUME + '.pdparams')
        model.set_dict(model_state)
        dino_loss_state = paddle.load(config.MODEL.RESUME + '._dino_loss.pdparams')
        criterion.set_dict(dino_loss_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        logger.info(
            f"----- Resume: Load model and optmizer from {config.MODEL.RESUME}")

    # STEP 7: Validation (eval mode)
    #if config.EVAL:
    #    logger.info('----- Start Validating')
    #    val_loss, val_acc1, val_acc5, val_time = validate(
    #        dataloader=dataloader_val,
    #        model=model,
    #        criterion=criterion,
    #        total_batch=len(dataloader_val),
    #        debug_steps=config.REPORT_FREQ,
    #        logger=logger)
    #    logger.info(f"Validation Loss: {val_loss:.4f}, " +
    #                f"Validation Acc@1: {val_acc1:.4f}, " +
    #                f"Validation Acc@5: {val_acc5:.4f}, " +
    #                f"time: {val_time:.2f}")
    #    return

    # STEP 8: Start training and validation (train mode)
    logger.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logger.info(f"Now training epoch {epoch}. LR={optimizer.get_lr():.6f}")
        train_loss, train_acc, train_time = train(dataloader=dataloader_train,
                                                  student_model=student_model,
                                                  teacher_model=teacher_model,
                                                  criterion=criterion,
                                                  optimizer=optimizer,
                                                  lr_schedule=lr_schedule,
                                                  wd_schedule=wd_schedule,
                                                  momentum_schedule=momentum_schedule,
                                                  freeze_last_layer=config.TRAIN.FREEZE_LAST_LAYER,
                                                  epoch=epoch,
                                                  total_epochs=config.TRAIN.NUM_EPOCHS,
                                                  total_batch=len(dataloader_train),
                                                  debug_steps=config.REPORT_FREQ,
                                                  accum_iter=config.TRAIN.ACCUM_ITER,
                                                  amp=config.AMP,
                                                  logger=logger)
        scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"Train Acc: {train_acc:.4f}, " +
                    f"time: {train_time:.2f}")
        ## validation
        #if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
        #    logger.info(f'----- Validation after Epoch: {epoch}')
        #    val_loss, val_acc1, val_acc5, val_time = validate(
        #        dataloader=dataloader_val,
        #        model=model,
        #        criterion=criterion,
        #        total_batch=len(dataloader_val),
        #        debug_steps=config.REPORT_FREQ,
        #        logger=logger)
        #    logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
        #                f"Validation Loss: {val_loss:.4f}, " +
        #                f"Validation Acc@1: {val_acc1:.4f}, " +
        #                f"Validation Acc@5: {val_acc5:.4f}, " +
        #                f"time: {val_time:.2f}")
        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(
                config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
            paddle.save(model.state_dict(), model_path + '.pdparams')
            paddle.save(optimizer.state_dict(), model_path + '.pdopt')
            paddle.save(criterion.state_dict(), model_path + '_dino_loss.pdprams')
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save loss: {model_path}_dino_loss.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")


if __name__ == "__main__":
    main()
