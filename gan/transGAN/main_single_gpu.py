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

"""transGAN training/validation using single GPU """

import sys
import os
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn

import paddle.nn.functional as F
from datasets_paddle import get_dataloader
from datasets_paddle import get_dataset
from utils import AverageMeter, WarmupCosineScheduler
from config import get_config, update_config

from models.ViT_custom import Generator
from models.ViT_custom_scale2 import Discriminator

parser = argparse.ArgumentParser('transGAN')
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
args = parser.parse_args()

# log format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")

# get default config
config = get_config()
# update config by arguments
config = update_config(config, args)
config.freeze()

# set output folder
# if not config.EVAL:
#     config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
# else:
#     config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

# set logging format
logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')

def train(args,
          gen_net,
          dis_net,
          gen_optimizer,
          dis_optimizer,
          fixed_z,
          lr_schedulers,
          dataloader,
          epoch,
          total_batch,
          debug_steps=2,
          accum_iter=1):
    """Training for one epoch
    Args:
        args: the default set of net
        gen_net: nn.Layer, the generator net
        dis_net: nn.Layer, the discriminator net
        gen_optimizer: generator's optimizer
        dis_optimizer: discriminator's optimizer
        fixed_z: the noise
        dataloader: paddle.io.DataLoader, dataloader instance
        lr_schedulers： learning rate
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_time
    """
    gen_net.train()
    dis_net.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        image = data[0]

        # Sample noise as generator input
        z = paddle.to_tensor(np.random.normal(0, 1, (image.shape[0], config.LATENT_DIM)))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        dis_optimizer.clear_grad()
        real_validity = dis_net(image)
        fake_imgs = gen_net(z, epoch).detach()
        fake_validity = dis_net(fake_imgs)

        d_loss = 0
        d_loss = paddle.mean(nn.ReLU()(1.0 - real_validity)) + \
                paddle.mean(nn.ReLU()(1 + fake_validity))

        #NOTE: division may be needed depending on the loss function
        # Here no division is needed:
        # default 'reduction' param in nn.CrossEntropyLoss is set to 'mean'
        d_loss = d_loss / accum_iter
        d_loss.backward()

        dis_optimizer.step()
        dis_optimizer.clear_grad()

        batch_size = image.shape[0]
        train_loss_meter.update(d_loss.numpy()[0], batch_size)

        # -----------------
        #  Train Generator
        # -----------------

        if epoch % 4 == 0:
            gen_optimizer.clear_grad()
            gen_z = paddle.to_tensor(np.random.normal(0, 1, (args.GEN_BATCH_SIZE, args.LATENT_DIM)))
            gen_imgs = gen_net(gen_z, epoch)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            loss_lz = paddle.to_tensor(0)
            g_loss = -paddle.mean(fake_validity)
            g_loss.backward()

            gen_optimizer.step()
            gen_optimizer.clear_grad()

            batch_size = image.shape[0]
            train_loss_meter.update(d_loss.numpy()[0] + g_loss.numpy()[0], batch_size)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Avg Loss: {train_loss_meter.avg:.4f}, ")

            # plt.figure()
            # for i in range(1,5):
            #     plt.subplot(2,2,i)
            #     plt.imshow(gen_imgs[i-1])
            #     plt.xticks([])
            #     plt.yticks([])
            # plt.draw()
            # plt.savefig(str(batch_id) + '.png')

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time


def main():
    # 0. Preparation
    last_epoch = config.TRAIN.LAST_EPOCH

    # 1. Create model
    gen_net = Generator(args=config)
    dis_net = Discriminator(args=config)
    gen_net = paddle.DataParallel(gen_net)
    dis_net = paddle.DataParallel(dis_net)
    print("import net end")

    # 2. Create train and val dataloader
    dataset_train = get_dataset(config, mode='train')
    # dataset_val = get_dataset(config, mode='train')
    dataloader_train = get_dataloader(config, dataset_train, 'train', False)
    # dataloader_val = get_dataloader(config, dataset_val, 'train', False)

    # 4. Define lr_scheduler
    gen_scheduler = None
    dis_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == "warmupcosine":
        gen_scheduler = WarmupCosineScheduler(learning_rate=config.TRAIN.BASE_LR,
                                              warmup_start_lr=config.TRAIN.WARMUP_START_LR,
                                              start_lr=config.TRAIN.BASE_LR,
                                              end_lr=config.TRAIN.END_LR,
                                              warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
                                              total_epochs=config.TRAIN.NUM_EPOCHS,
                                              last_epoch=config.TRAIN.LAST_EPOCH,
                                             )
        dis_scheduler = WarmupCosineScheduler(learning_rate=config.TRAIN.BASE_LR,
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
    else:
        logging.fatal(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")
        raise NotImplementedError(f"Unsupported Scheduler: {config.TRAIN.LR_SCHEDULER}.")

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
        optimizer_gen = paddle.optimizer.AdamW(
            parameters=gen_net.parameters(),
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            epsilon=config.TRAIN.OPTIMIZER.EPS)
        optimizer_dis = paddle.optimizer.AdamW(
            parameters=dis_net.parameters(),
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            epsilon=config.TRAIN.OPTIMIZER.EPS)
    else:
        logging.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # initial
    fixed_z = paddle.to_tensor(np.random.normal(0, 1, (100, config.LATENT_DIM)))
    start_epoch = 0
    best_fid = 1e4

    # 6. Load pretrained model or load resume model and optimizer states
    # if config.MODEL.PRETRAINED:
    #     assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams')
    #     model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
    #     model.set_dict(model_state)
    #     logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    # if config.MODEL.RESUME and os.path.isfile(
    #         config.MODEL.RESUME+'.pdparams') and os.path.isfile(
    #             config.MODEL.RESUME+'.pdopt'):
    #     model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
    #     model.set_dict(model_state)
    #     opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
    #     optimizer.set_state_dict(opt_state)
    #     logger.info(
    #         "----- Resume: Load model and optmizer from {config.MODEL.RESUME}")

    # 7. Validation
    if config.EVAL:
        logger.info('----- Start Validating')
        val_loss, val_acc1, val_acc5, val_time = validate(
            dataloader=dataloader_val,
            model=model,
            criterion=criterion,
            total_batch=len(dataloader_val),
            debug_steps=config.REPORT_FREQ)
        logger.info(f"Validation Loss: {val_loss:.4f}, " +
                    f"Validation Acc@1: {val_acc1:.4f}, " +
                    f"Validation Acc@5: {val_acc5:.4f}, " +
                    f"time: {val_time:.2f}")
        return
    # 8. Start training and validation
    logging.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        lr_schedulers = (gen_scheduler, dis_scheduler) if config.LR_DECAY else None

        logging.info(f"Now training epoch {epoch}. LR={optimizer_dis.get_lr():.6f}")
        logging.info(f"Now training epoch {epoch}. LR={optimizer_gen.get_lr():.6f}")

        train_loss, train_time = train(config,
                                       gen_net,
                                       dis_net,
                                       optimizer_gen,
                                       optimizer_dis,
                                       fixed_z,
                                       lr_schedulers,
                                       dataloader=dataloader_train,
                                       epoch=epoch,
                                       total_batch=1000,
                                       debug_steps=config.REPORT_FREQ,
                                       accum_iter=config.TRAIN.ACCUM_ITER,
                                      )
        # lr_schedulers.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        # if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
        #     logger.info(f'----- Validation after Epoch: {epoch}')
        #     val_loss, val_acc1, val_acc5, val_time = validate(
        #         dataloader=dataloader_val,
        #         model=model,
        #         criterion=criterion,
        #         total_batch=len(dataloader_val),
        #         debug_steps=config.REPORT_FREQ)
        #     logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
        #                 f"Validation Loss: {val_loss:.4f}, " +
        #                 f"Validation Acc@1: {val_acc1:.4f}, " +
        #                 f"Validation Acc@5: {val_acc5:.4f}, " +
        #                 f"time: {val_time:.2f}")
        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(
                config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
            paddle.save(dis_net.state_dict(), model_path)
            paddle.save(gen_net.state_dict(), model_path)
            paddle.save(optimizer.state_dict(), model_path)
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")


if __name__ == "__main__":
    main()
