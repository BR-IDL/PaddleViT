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

"""MAE pretraining using multiple GPU """

import sys
import os
import time
import argparse
import random
import math
import numpy as np
import paddle
from paddle.distributed import fleet
from datasets import get_dataloader
from datasets import get_dataset
from config import get_config
from config import update_config
from utils import AverageMeter
from utils import get_logger
from utils import write_log
from utils import all_reduce_mean
from utils import skip_weight_decay_fn
from utils import get_params_groups
from utils import adjust_learning_rate
#from mixup import Mixup
#from losses import LabelSmoothingCrossEntropyLoss
#from losses import SoftTargetCrossEntropyLoss
from transformer import build_mae_pretrain as build_model
import paddlenlp

def get_arguments():
    """return argumeents, this will overwrite the config by (1) yaml file (2) argument values"""
    parser = argparse.ArgumentParser('MAE Pretrain')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-output', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-batch_size_eval', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=None)
    parser.add_argument('-accum_iter', type=int, default=None)
    parser.add_argument('-pretrained', type=str, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-amp', action='store_true')
    arguments = parser.parse_args()
    return arguments


def train(dataloader,
          model,
          mask_ratio,
          optimizer,
          lr_scheduler,
          base_lr,
          min_lr,
          epoch,
          warmup_epochs,
          total_epochs,
          total_batches,
          debug_steps=100,
          accum_iter=1,
          amp_grad_scaler=None,
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
        total_batches: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        accum_iter: int, num of iters for accumulating gradients, default: 1
        amp_grad_scaler: GradScaler/None, if not None, pass the GradScaler and enable AMP training, default: None
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

    #if amp is True:
    #    scaler = paddle.amp.GradScaler() # default init_loss_scaling = 32768
    optimizer.clear_grad()

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        batch_size = images.shape[0]
        # adjust learning rate
        if batch_id % accum_iter == 0:
            lr_scheduler.step(batch_id / total_batches + epoch -1)
            #adjust_learning_rate(optimizer,
            #                     base_lr,
            #                     min_lr,
            #                     batch_id / total_batches + epoch - 1, 
            #                     warmup_epochs, 
            #                     total_epochs)
        # forward
        with paddle.amp.auto_cast(amp_grad_scaler is not None):
            loss, _, _ = model(images)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        # backward and step
        if amp_grad_scaler is None: # fp32
            loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.clear_grad()
        else: # amp
            scaled_loss = amp_grad_scaler.scale(loss)
            scaled_loss.backward()
            if ((batch_id + 1) % accum_iter == 0) or (batch_id + 1 == len(dataloader)):
                # amp for param group reference: https://github.com/PaddlePaddle/Paddle/issues/37188
                amp_grad_scaler.step(optimizer)
                amp_grad_scaler.update()
                optimizer.clear_grad()

        # sync from other gpus for overall loss and acc
        master_loss = all_reduce_mean(loss_value)
        master_batch_size = all_reduce_mean(batch_size)
        master_loss_meter.update(master_loss, master_batch_size)
        train_loss_meter.update(loss_value, batch_size)
        if batch_id % debug_steps == 0 or batch_id + 1 == len(dataloader):
            general_message = (f"Epoch[{epoch:03d}/{total_epochs:03d}], "
                               f"Step[{batch_id:04d}/{total_batches:04d}], "
                               f"Lr: {optimizer.get_lr():.6e}, ")
            local_message = (general_message +
                             f"Loss: {loss_value:.4f} ({train_loss_meter.avg:.4f})")
            master_message = (general_message +
                              f"Loss: {master_loss:.4f} ({master_loss_meter.avg:.4f})")
            write_log(local_logger, master_logger, local_message, master_message)

    paddle.distributed.barrier()
    train_time = time.time() - time_st
    return train_loss_meter.avg, master_loss_meter.avg, train_time


def main_worker(*args):
    """main method for each process"""
    # STEP 0: Preparation
    paddle.device.set_device('gpu')
    #paddle.distributed.init_parallel_env()
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    config = args[0]
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # logger for each process/gpu
    local_logger, master_logger = get_logger(config.SAVE)
    message = (f'----- world_size = {world_size}, local_rank = {local_rank} \n'
               f'----- {config}')
    write_log(local_logger, master_logger, message)

    # STEP 1: Create model
    model = build_model(config)
    if paddle.distributed.get_world_size() > 1:
        strategy = fleet.DistributedStrategy()
        ## Hybrid Parallel Training
        strategy.hybrid_configs = {}
        fleet.init(is_collective=True, strategy=strategy)

    # STEP 2: Create train dataloader
    dataset_train = args[1]
    dataloader_train = get_dataloader(config, dataset_train, True, True)
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
    # define scaler for amp training
    amp_grad_scaler = paddle.amp.GradScaler() if config.AMP else None 
    # set gradient clip
    if config.TRAIN.GRAD_CLIP:
        clip = paddle.nn.ClipGradByGlobalNorm(config.TRAIN.GRAD_CLIP)
    else:
        clip = None
    # set optimizer
	# create warmup and cosine decay lr scheduler
    if config.TRAIN.WARMUP_EPOCHS > 0:
        cosine_lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=config.TRAIN.BASE_LR,
            T_max=config.TRAIN.NUM_EPOCHS - config.TRAIN.WARMUP_EPOCHS,
            eta_min=config.TRAIN.END_LR,
            last_epoch=-1) # do not set last epoch, handled in warmup sched get_lr()
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=cosine_lr_scheduler, # use cosine lr sched after warmup
            warmup_steps=config.TRAIN.WARMUP_EPOCHS, # only support position integet
            start_lr=config.TRAIN.WARMUP_START_LR,
            end_lr=config.TRAIN.BASE_LR,
            last_epoch=config.TRAIN.LAST_EPOCH)
    else: # create cosine decay lr scheduler if no warmup epochs
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=config.TRAIN.BASE_LR,
            T_max=config.TRAIN.NUM_EPOCHS,
            eta_min=config.TRAIN.END_LR,
            last_epoch=config.TRAIN.LAST_EPOCH)

    if config.TRAIN.OPTIMIZER.NAME == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=lr_scheduler, # now only support warmup + consine 
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            grad_clip=clip,
            apply_decay_param_fun=skip_weight_decay_fn(
                model, # skip bn and bias in model
                ['encoder_position_embedding', 'cls_token']), # skip custom ops
        )
    elif config.TRAIN.OPTIMIZER.NAME == "AdamWDL":  # using paddlenlp's impl
        optimizer = paddlenlp.ops.optimizer.AdamWDL(
            learning_rate=lr_scheduler,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            layerwise_decay=config.TRAIN.LAYER_DECAY,
            n_layers=config.MODEL.ENCODER.DEPTH,
            set_param_lr_fun=lr_decay.lr_setting,
            parameters=model.parameters(),
            name_dict=name_dict,
            apply_decay_param_fun=skip_weight_decay_fn(
                model, # skip bn and bias in model
                ['encoder_position_embedding', 'cls_token']), # skip custom ops
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
        assert os.path.isfile(config.MODEL.PRETRAINED) is True
        model_state = paddle.load(config.MODEL.PRETRAINED)
        if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
            # pretrain only load model weight, opt and epoch are ignored
            model.set_state_dict(model_state['model'])
        else: # direct load pdparams without other items
            model.set_state_dict(model_state)
        message = f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}"
        write_log(local_logger, master_logger, message)

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME) is True
        model_state = paddle.load(config.MODEL.RESUME)
        if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
            model.set_state_dict(model_state['model'])
            if 'optimizer' in model_state:
                optimizer.set_state_dict(model_state['optimizer'])
            if 'lr_scheduler' in model_state and lr_scheduler is not None:
                lr_scheduler.set_state_dict(model_state['lr_scheduler'])
            if 'epoch' in model_state:
                config.TRAIN.LAST_EPOCH = model_state['epoch']
            if 'amp_grad_scaler' in model_state and amp_grad_scaler is not None:
                amp_grad_scaler.load_state_dict(model_state['amp_grad_scaler'])
            if config.TRAIN.MODEL_EMA:
                model_ema.module.set_state_dict(model_state['model_ema'])
            lr_scheduler.step(config.TRAIN.LAST_EPOCH)
            message = (f"----- Resume Training: Load model from {config.MODEL.RESUME}, "
                       f"opt = [{'optimizer' in model_state}], "
                       f"lr_scheduler = [{'lr_scheduler' in model_state}], "
                       f"model_ema = [{'model_ema' in model_state}], "
                       f"epoch = [{model_state.get('epoch', -1)}], "
                       f"amp_grad_scaler = [{'amp_grad_scaler' in model_state}]")
            write_log(local_logger, master_logger, message)
        else: # direct load pdparams without other items
            message = f"----- Resume Training: Load from {config.MODEL.RESUME}, no opt/epoch/scaler"
            write_log(local_logger, master_logger, message, 'warning')
            model.set_state_dict(model_state)

    write_log(local_logger, master_logger, f"----- Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, config.TRAIN.NUM_EPOCHS + 1):
        # Train one epoch
        write_log(local_logger, master_logger, f"Train epoch {epoch}. LR={optimizer.get_lr():.6e}")

        train_loss, avg_loss, train_time = train(
            dataloader=dataloader_train,
            model=model,
            mask_ratio=config.MODEL.MASK_RATIO,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            base_lr=config.TRAIN.BASE_LR,
            min_lr=config.TRAIN.END_LR,
            epoch=epoch,
            warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
            total_epochs=config.TRAIN.NUM_EPOCHS,
            total_batches=total_batch_train,
            debug_steps=config.REPORT_FREQ,
            accum_iter=config.TRAIN.ACCUM_ITER,
            amp_grad_scaler=amp_grad_scaler,
            local_logger=local_logger,
            master_logger=master_logger)

        general_message = (f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], "
                           f"Lr: {optimizer.get_lr():.6e}, "
                           f"time: {train_time:.2f}, ")
        local_message = (general_message +
                         f"Train Loss: {train_loss:.4f}")
        master_message = (general_message +
                          f"Train Loss: {avg_loss:.4f}")
        write_log(local_logger, master_logger, local_message, master_message)

        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(
                    config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{avg_loss}.pdparams")
                state_dict = dict()
                state_dict['model'] = model.state_dict()
                state_dict['optimizer'] = optimizer.state_dict()
                state_dict['epoch'] = epoch
                if amp_grad_scaler is not None:
                    state_dict['amp_grad_scaler'] = amp_grad_scaler.state_dict()
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                paddle.save(state_dict, model_path)
                message = (f"----- Save model: {model_path}")
                write_log(local_logger, master_logger, message)


def main():
    # config is updated in order: (1) default in config.py, (2) yaml file, (3) arguments
    config = update_config(get_config(), get_arguments())
    # set output folder
    config.SAVE = '{}/pretrain-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M'))
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)
    # get dataset
    dataset_train = get_dataset(config, is_train=True)
    # start training
    #paddle.distributed.spawn(main_worker, args=(config, dataset_train, ))
    main_worker(config, dataset_train, )


if __name__ == "__main__":
    main()
