#!/usr/bin/python3
import os
import time
import random
import argparse
import numpy as np
from collections import deque
import paddle
import paddle.nn as nn
from config import *
from src.utils import get_sys_env, logger
from src.datasets import get_dataset
from src.models import get_model
from src.transforms import *
from src.models.losses import CrossEntropyLoss
from src.utils import TimeAverager, calculate_eta, resume


def parse_args():
    parser = argparse.ArgumentParser(description='Visual Transformer for semantic segmentation')
    parser.add_argument("--config", dest='cfg',help="The config file.", default=None, type=str)
    return parser.parse_args()

def optimizer_setting(model, config):
    if config.TRAIN.LR_SCHEDULER.NAME == "PolynomialDecay":
        scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=config.TRAIN.BASE_LR, 
            decay_steps=config.TRAIN.ITERS, 
            end_lr=config.TRAIN.END_LR, 
            power=config.TRAIN.POWER, 
            cycle=False, 
            last_epoch=-1, 
            verbose=False)
    else:
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=model.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM)
    else:
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
    return optimizer


def multi_cross_entropy_loss(pred_list, label, num_classes=60, weights=[1, 0.4, 0.4, 0.4, 0.4]):
    label = paddle.reshape(label, [-1, 1]) # (b, h, w) -> (bhw, 1)                                      
    label.stop_gradient = True
    loss = 0
    for i in range(len(pred_list)):
        pred_i = paddle.transpose(pred_list[i], perm=[0, 2, 3, 1]) # (b,c,h,w) -> (b,h,w,c)
        pred_i = paddle.reshape(pred_i, [-1, num_classes])# (b,h,w,c) -> (bhw, c)
        pred_i = nn.functional.softmax(pred_i, axis=1)  
        loss_i = nn.functional.cross_entropy(pred_i, label, ignore_index=255)
        loss += weights[i]*loss_i
    return loss


def main():
    config = get_config()
    args = parse_args()
    config = update_config(config, args)
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
    paddle.set_device(place)
    # build  model
    model = get_model(config)
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    # build optimizer
    optimizer = optimizer_setting(model, config)
    # build dataset_train
    transforms_train = [ 
        ResizeStepScaling(min_scale_factor = 0.5, max_scale_factor = 2.0, scale_step_size = 0.25),
        RandomPaddingCrop(crop_size = config.DATA.CROP_SIZE, 
            im_padding_value = (123.675, 116.28, 103.53), 
            label_padding_value = 255),
        RandomHorizontalFlip(prob = 0.5),
        RandomDistort(brightness_range = 0.4, contrast_range = 0.4, saturation_range = 0.4),
        Normalize(mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375])
    ]
    dataset_train = get_dataset(config, data_transform = transforms_train, mode = 'train')

    batch_sampler = paddle.io.DistributedBatchSampler(
        dataset_train, batch_size=config.DATA.BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        return_list=True,
    )
    logger.info("train_loader.len= {}".format(len(train_loader)))
    start_iter = 0
    # TODO: Resume from checkpoint, and update start_iter
    # build workspace for saving checkpoints
    if not os.path.isdir(config.SAVE_DIR):
        if os.path.exists(config.SAVE_DIR):
            os.remove(config.SAVE_DIR)
        os.makedirs(config.SAVE_DIR)

    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
            logger.info("using dist training")
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    avg_loss = 0.0
    avg_loss_list = []
    iters_per_epoch = len(batch_sampler)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    cur_iter = start_iter
    # begin training
    while cur_iter < config.TRAIN.ITERS:
        for data in train_loader:
            cur_iter += 1
            if cur_iter > config.TRAIN.ITERS:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            labels = data[1].astype('int64')
            if nranks > 1:
                logits_list = ddp_model(images)
            else:
                logits_list = model(images)
            loss_list = multi_cross_entropy_loss(logits_list, labels, num_classes = config.DATA.NUM_CLASSES)
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            lr = optimizer.get_lr()
            if isinstance(optimizer._learning_rate,paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
                #print("updating lr, lr_last={}, lr_cur={}".format(lr, optimizer.get_lr()))
                #print("decay_steps: {}, last_epoch: {}".format(optimizer._learning_rate.decay_steps, 
                #    optimizer._learning_rate.last_epoch))
            model.clear_gradients()
            avg_loss += loss.numpy()[0]
            if not avg_loss_list:
                avg_loss_list = [l.numpy() for l in loss_list]
            else:
                for i in range(len(loss_list)):
                    avg_loss_list[i] += loss_list[i].numpy()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=config.DATA.BATCH_SIZE)
            if (cur_iter) % config.LOGGING_INFO_FREQ == 0 and local_rank == 0:
                avg_loss /= config.LOGGING_INFO_FREQ
                avg_loss_list = [l[0] / config.LOGGING_INFO_FREQ for l in avg_loss_list]
                remain_iters = config.TRAIN.ITERS - cur_iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info("[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.8f}, batch_cost:\
                    {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}".format(
                    (cur_iter - 1) // iters_per_epoch + 1, cur_iter, config.TRAIN.ITERS, avg_loss, 
                    lr, avg_train_batch_cost, avg_train_reader_cost, 
                    batch_cost_averager.get_ips_average(), eta))
                avg_loss = 0.0
                avg_loss_list = []
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (cur_iter % config.SAVE_FREQ_CHECKPOINT == 0 or cur_iter == config.TRAIN.ITERS) and local_rank == 0:
                current_save_weigth_file = os.path.join(config.SAVE_DIR,"iter_{}_model_state.pdparams".format(cur_iter))
                current_save_opt_file = os.path.join(config.SAVE_DIR,"iter_{}_opt_state.pdopt".format(cur_iter))
                paddle.save(model.state_dict(), current_save_weigth_file)
                paddle.save(optimizer.state_dict(), current_save_opt_file)
                save_models.append([current_save_weigth_file,current_save_opt_file])
                logger.info("saving the weights of model to {}".format(current_save_weigth_file))
                if len(save_models) > config.KEEP_CHECKPOINT_MAX > 0:
                    files_to_remove = save_models.popleft()
                    os.remove(files_to_remove[0])
                    os.remove(files_to_remove[1])
            batch_start = time.time()
    time.sleep(0.5)

if __name__ == '__main__':
    main()
