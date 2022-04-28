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

"""Swin test using multiple GPU"""
import sys
import os
import time
import argparse
import random
import math
import numpy as np
import paddle
from datasets import get_dataloader
from datasets import get_dataset
from config import get_config
from config import update_config
from utils import AverageMeter
from utils import get_logger
from utils import write_log
from utils import all_reduce_mean
from interpolate_position_embedding import interpolate_position_embedding
from swin import build_swin as build_model


def get_arguments():
    """return argumeents, this will overwrite the config by (1) yaml file (2) argument values"""
    parser = argparse.ArgumentParser('Swin')
    parser.add_argument('-cfg', type=str, default=None)
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-data_folder', type=str, default=None)
    parser.add_argument('-anno_folder', type=str, default=None)
    parser.add_argument('-data_list_train', type=str, default=None)
    parser.add_argument('-data_list_val', type=str, default=None)
    parser.add_argument('-class_type', type=str, default=None)
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


@paddle.no_grad()
def validate(dataloader,
             model,
             criterion,
             total_batches,
             debug_steps=100,
             local_logger=None,
             master_logger=None,
             save='./'):
    """Validation for the whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batches: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        local_logger: logger for local process/gpu, default: None
        master_logger: logger for main process, default: None
    Returns:
        val_loss_meter.avg: float, average loss on current process/gpu
        val_acc1_meter.avg: float, average top1 accuracy on current processes/gpus
        master_loss_meter.avg: float, average loss on all processes/gpus
        master_acc1_meter.avg: float, average top1 accuracy on all processes/gpus
        val_time: float, validation time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    master_loss_meter = AverageMeter()
    master_acc1_meter = AverageMeter()

    time_st = time.time()

    # output path
    local_rank = paddle.distributed.get_rank()
    ofile = open(os.path.join(save, f'pred_{local_rank}.txt'), 'w')

    for batch_id, data in enumerate(dataloader):
        # get data
        images = data[0]
        label = data[1]
        image_path = data[2]
        batch_size = images.shape[0]

        output = model(images)
        if label is not None:
            loss = criterion(output, label)
            loss_value = loss.item()

        pred = paddle.nn.functional.softmax(output)

        if label is not None:
            acc1 = paddle.metric.accuracy(pred, label.unsqueeze(1)).item()

            # sync from other gpus for overall loss and acc
            master_loss = all_reduce_mean(loss_value)
            master_acc1 = all_reduce_mean(acc1)
            master_batch_size = all_reduce_mean(batch_size)

            master_loss_meter.update(master_loss, master_batch_size)
            master_acc1_meter.update(master_acc1, master_batch_size)
            val_loss_meter.update(loss_value, batch_size)
            val_acc1_meter.update(acc1, batch_size)

            if batch_id % debug_steps == 0:
                local_message = (f"Step[{batch_id:04d}/{total_batches:04d}], "
                                 f"Avg Loss: {val_loss_meter.avg:.4f}, "
                                 f"Avg Acc@1: {val_acc1_meter.avg:.4f}")
                master_message = (f"Step[{batch_id:04d}/{total_batches:04d}], "
                                  f"Avg Loss: {master_loss_meter.avg:.4f}, "
                                  f"Avg Acc@1: {master_acc1_meter.avg:.4f}")
                write_log(local_logger, master_logger, local_message, master_message)
        else:
            if batch_id % debug_steps == 0:
                local_message = f"Step[{batch_id:04d}/{total_batches:04d}]"
                master_message = f"Step[{batch_id:04d}/{total_batches:04d}]"
                write_log(local_logger, master_logger, local_message, master_message)

        # write results to pred
        for idx, img_p in enumerate(image_path):
            pred_prob, pred_label = paddle.topk(pred[idx], 1)
            pred_label = pred_label.cpu().numpy()[0]
            ofile.write(f'{img_p} {pred_label}\n')

    val_time = time.time() - time_st
    ofile.close()
    return (val_loss_meter.avg,
            val_acc1_meter.avg,
            master_loss_meter.avg,
            master_acc1_meter.avg,
            val_time)


def main_worker(*args):
    """main method for each process"""
    # STEP 0: Preparation
    paddle.device.set_device('gpu')
    paddle.distributed.init_parallel_env()
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    config = args[0]
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED + local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    local_logger, master_logger = get_logger(config.SAVE)
    message = (f'----- world_size = {world_size}, local_rank = {local_rank} \n'
               f'----- {config}')
    write_log(local_logger, master_logger, message)

    # STEP 1: Create model
    model = build_model(config)

    # STEP 2: load data
    dataset_val = args[1]
    dataloader_val = get_dataloader(config, dataset_val, False, True)
    total_batch_val = len(dataloader_val)
    message = f'----- Total # of val batch (single gpu): {total_batch_val}'
    write_log(local_logger, master_logger, message)

    # (Optional) Use CrossEntropyLoss for val
    criterion_val = paddle.nn.CrossEntropyLoss()

    # STEP 3: Load pretrained model weights
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED) is True
        model_state = paddle.load(config.MODEL.PRETRAINED)
        if 'model' in model_state: # load state_dict with multi items: model, optimier, and epoch
            # pretrain only load model weight, opt and epoch are ignored
            if 'model_ema' in model_state:
                model_state = model_state['model_ema']
            else:
                model_state = model_state['model']
        # delete relative_position_index since it is always re-initialized
        for key in [k for k in model_state.keys() if 'relative_position_index' in k]:
            del model_state[key]
        # delete relative_coords_table since it is always re-initialized
        for key in [k for k in model_state.keys() if 'relative_coords_table' in k]:
            del model_state[key]
        # delete attn_mask since it is always re-initialized
        for key in [k for k in model_state.keys() if 'attn_mask' in k]:
            del model_state[key]
        # interpolate pos tokens if num of model's tokens not equal to num of model_state's tokens
        interpolate_position_embedding(model, model_state)
        model.set_state_dict(model_state)
        message = f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}"
        write_log(local_logger, master_logger, message)
    else:
        message = f"----- Pretrained model not loaded: config.MODEL.PRETRAINED: {config.MODEL.PRETRAINED}"
        write_log(local_logger, master_logger, message, 'faltal')
        raise ValueError('Pretrained model none')

    # STEP 4: Enable model data parallelism on multi processes
    model = paddle.DataParallel(model)

    # STEP 5: Run testing / evaluation
    write_log(local_logger, master_logger, "----- Start Testing/Validation")
    val_loss, val_acc1, avg_loss, avg_acc1, val_time = validate(
        dataloader=dataloader_val,
        model=model,
        criterion=criterion_val,
        total_batches=total_batch_val,
        debug_steps=config.REPORT_FREQ,
        local_logger=local_logger,
        master_logger=master_logger,
        save=config.SAVE)
    local_message = ("----- Validation: " +
                     f"Validation Loss: {val_loss:.4f}, " +
                     f"Validation Acc@1: {val_acc1:.4f}, " +
                     f"time: {val_time:.2f}")
    master_message = ("----- Validation: " +
                     f"Validation Loss: {avg_loss:.4f}, " +
                     f"Validation Acc@1: {avg_acc1:.4f}, " +
                     f"time: {val_time:.2f}")
    write_log(local_logger, master_logger, local_message, master_message)


def main():
    # config is updated in order: (1) default in config.py, (2) yaml file, (3) arguments
    config = update_config(get_config(), get_arguments())

    # set output folder
    config.SAVE = os.path.join(config.SAVE,
        f"test-{time.strftime('%Y%m%d-%H-%M')}")
    if not os.path.exists(config.SAVE):
        os.makedirs(config.SAVE, exist_ok=True)

    # get test/val dataset
    dataset = get_dataset(config, is_train=False)

    # dist spawn lunch: use CUDA_VISIBLE_DEVICES to set available gpus
    #paddle.distributed.spawn(main_worker, args=(config, dataset))
    main_worker(config, dataset)


if __name__ == "__main__":
    main()
