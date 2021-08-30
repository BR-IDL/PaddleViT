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

"""Styleformer training/validation using single GPU """

import sys
import os
import time
import logging
import argparse
import random
import numpy as np
import paddle
from datasets import get_dataloader
from datasets import get_dataset
from generator import Generator
from discriminator import StyleGANv2Discriminator
from utils.utils import AverageMeter
from utils.utils import WarmupCosineScheduler
from utils.utils import gradient_penalty
from config import get_config
from config import update_config
from metrics.fid import FID

parser = argparse.ArgumentParser('Styleformer')
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


log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")

# get default config
config = get_config()
# update config by arguments
config = update_config(config, args)

# set output folder
if not config.EVAL:
    config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
else:
    config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

config.freeze()

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

# set logging format
logger = logging.getLogger()
file_handler = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)
logger.info(f'config= {config}')

def train(dataloader,
          gen,
          dis,
          gen_optimizer,
          dis_optimizer,
          epoch,
          total_batch,
          debug_steps=100):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """
    gen.train()
    train_loss_meter = AverageMeter()
    time_st = time.time()
    lambda_gp = 10
    # fid = FID()
    for batch_id, data in enumerate(dataloader):
        dis_optimizer.clear_grad()
        real_img = data[0]
        batch_size = real_img.shape[0]

        noise = paddle.randn([batch_size, gen.z_dim])
        fake_img = gen(noise, c=paddle.zeros([0]))
        fake_img = (fake_img * 127.5 + 128).clip(0, 255).astype('uint8')
        fake_img = fake_img / 255.0
        fake_pred = dis(fake_img.detach())
        real_pred = dis(real_img)

        # fid.update(fake_img, real_img)
        # fid_score = fid.accumulate()
        # print(fake_pred[0],real_pred[0])
        gp = gradient_penalty(dis, real_img, fake_img.detach())
        d_loss = -(paddle.mean(real_pred) - paddle.mean(fake_pred)) + lambda_gp * gp

        d_loss.backward()
        dis_optimizer.step()

        for _ in range(5):
            gen_optimizer.clear_grad()
            noise = paddle.randn([batch_size, gen.z_dim])
            gen_img = gen(noise, c=paddle.zeros([0]))
            gen_img = (gen_img * 127.5 + 128).clip(0, 255).astype('uint8')
            gen_img = gen_img / 255.0
            #gen_imgs=paddle.multiply(gen_img,paddle.to_tensor(127.5))
            #gen_imgs=paddle.clip(paddle.add(
            #   gen_imgs,paddle.to_tensor(127.5)).transpose((0,2,3,1)),
            #   min=0.0,max=255.0).astype('uint8')

            fake_pred = dis(gen_img)
            g_loss = -paddle.mean(fake_pred)

            g_loss.backward()
            gen_optimizer.step()

        train_loss_meter.update(d_loss.numpy()[0] + g_loss.numpy()[0], batch_size)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"D Loss: {d_loss.item():.4f}, " +
                f"G Loss: {g_loss.item():.4f}")

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time

def r1_penalty(real_pred, real_img):
    """
    R1 regularization for discriminator. The core idea is to
    penalize the gradient on real data alone: when the
    generator distribution produces the true data distribution
    and the discriminator is equal to 0 on the data manifold, the
    gradient penalty ensures that the discriminator cannot create
    a non-zero gradient orthogonal to the data manifold without
    suffering a loss in the GAN game.
    Ref:
    Eq. 9 in Which training methods for GANs do actually converge.
    """

    grad_real = paddle.grad(outputs=real_pred.sum(),
                            inputs=real_img,
                            create_graph=True)[0]
    grad_penalty = (grad_real * grad_real).reshape([grad_real.shape[0],
                                                    -1]).sum(1).mean()
    return grad_penalty


def validate(dataloader,
             model,
             batch_size,
             total_batch,
             num_classes,
             max_real_num=None,
             max_gen_num=None,
             debug_steps=32):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        batch_size: int, batch size (used to init FID measturement)
        total_epoch: int, total num of epoch, for logging
        max_real_num: int, max num of real images loaded from dataset
        max_gen_num: int, max num of fake images genearted for validation
        debug_steps: int, num of iters to log info
    Returns:
        fid_score: float, fid score
        val_time: int, validation time in ms
    """
    model.eval()
    time_st = time.time()
    fid = FID(batch_size)
    fid_preds_all = []
    fid_gts_all = []
    # similar to metric type: fid50k_full, fid50k, etc.
    if max_real_num is not None:
        max_real_batch = max_real_num // batch_size
    else:
        max_real_batch = total_batch
    if max_gen_num is not None:
        max_gen_batch = max_gen_num // batch_size
    else:
        max_gen_batch = total_batch

    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            if batch_id >= max_real_batch:
                break
            curr_batch_size = data[0].shape[0]
            fid.batch_size = curr_batch_size

            real_image = data[0]
            z = paddle.randn([curr_batch_size, model.z_dim])
            fake_image = model(z, c=paddle.randint(0, num_classes, [curr_batch_size]))

            fake_image = (fake_image * 127.5 + 128).clip(0, 255).astype('uint8')
            fake_image = fake_image / 255.0

            fid.update(fake_image, real_image)

            if batch_id < max_gen_batch:
                fid_preds_all.extend(fid.preds)
            fid_gts_all.extend(fid.gts)
            fid.reset()
            if batch_id % debug_steps == 0:
                if batch_id >= max_gen_batch:
                    logger.info(f"Val Step[{batch_id:04d}/{total_batch:04d}] done (no gen)")
                else:
                    logger.info(f"Val Step[{batch_id:04d}/{total_batch:04d}] done")

    fid.preds = fid_preds_all
    fid.gts = fid_gts_all
    fid_score = fid.accumulate()
    val_time = time.time() - time_st
    return fid_score, val_time


def main():
    """main function for training and validation"""
    # 0. Preparation
    last_epoch = config.TRAIN.LAST_EPOCH
    seed = config.SEED
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Create model
    gen = Generator(config)
    dis = StyleGANv2Discriminator(config)

    # 2. Create train and val dataloader
    dataset_train = get_dataset(config, mode='train')
    dataset_val = get_dataset(config, mode='val')
    dataloader_train = get_dataloader(config, dataset_train, 'train', False)
    dataloader_val = get_dataloader(config, dataset_val, 'val', False)

    # 3. Define criterion
    # validation criterion (FID) is defined in validate method

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
        gen_optimizer = paddle.optimizer.Momentum(
            parameters=gen.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
        dis_optimizer = paddle.optimizer.Momentum(
            parameters=dis.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            grad_clip=clip)
    elif config.TRAIN.OPTIMIZER.NAME == "Adam":
        gen_optimizer = paddle.optimizer.Adam(
            parameters=gen.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            )
        dis_optimizer = paddle.optimizer.Adam(
            parameters=dis.parameters(),
            learning_rate=scheduler if scheduler is not None else config.TRAIN.BASE_LR,
            beta1=config.TRAIN.OPTIMIZER.BETAS[0],
            beta2=config.TRAIN.OPTIMIZER.BETAS[1],
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            epsilon=config.TRAIN.OPTIMIZER.EPS,
            )
    else:
        logging.fatal(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")
        raise NotImplementedError(f"Unsupported Optimizer: {config.TRAIN.OPTIMIZER.NAME}.")

    # 6. Load pretrained model or load resume model and optimizer states
    if config.MODEL.PRETRAINED:
        assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams')
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
        gen.set_dict(model_state["gen_state_dict"])
        dis.set_dict(model_state["dis_state_dict"])
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME:
        assert os.path.isfile(config.MODEL.RESUME + '.pdparams') is True
        assert os.path.isfile(config.MODEL.RESUME + '.pdopt') is True
        # load model weights
        model_state = paddle.load(config.MODEL.RESUME + '.pdparams')
        gen.set_dict(model_state["gen_state_dict"])
        dis.set_dict(model_state["dis_state_dict"])
        # load optimizer
        opt_state = paddle.load(config.MODEL.RESUME + '.pdopt')
        gen_optimizer.set_state_dict(opt_state["gen_state_dict"])
        dis_optimizer.set_state_dict(opt_state["dis_state_dict"])
        logger.info(f"----- Resume: Load model and optmizer from {config.MODEL.RESUME}")

    # 7. Validation
    if config.EVAL:
        logger.info('----- Start Validating')
        fid_score, val_time = validate(
            dataloader=dataloader_val,
            model=gen,
            batch_size=config.DATA.BATCH_SIZE,
            total_batch=len(dataloader_val),
            num_classes=config.MODEL.NUM_CLASSES,
            max_real_num=config.DATA.MAX_REAL_NUM,
            max_gen_num=config.DATA.MAX_GEN_NUM,
            debug_steps=config.REPORT_FREQ)
        logger.info(f" ----- FID: {fid_score:.4f}, time: {val_time:.2f}")
        return

    # 8. Start training and validation
    logging.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        logging.info(f"Now training epoch {epoch}. LR={gen_optimizer.get_lr():.6f}")
        train_loss, train_time = train(dataloader=dataloader_train,
                                       gen=gen,
                                       dis=dis,
                                       gen_optimizer=gen_optimizer,
                                       dis_optimizer=dis_optimizer,
                                       epoch=epoch,
                                       total_batch=len(dataloader_train),
                                       debug_steps=config.REPORT_FREQ)
        scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            logger.info(f'----- Validation after Epoch: {epoch}')
            fid_score, val_time = validate(
                dataloader=dataloader_val,
                model=gen,
                batch_size=config.DATA.BATCH_SIZE,
                total_batch=len(dataloader_val),
                num_classes=config.MODEL.NUM_CLASSES,
                max_real_num=config.DATA.MAX_REAL_NUM,
                max_gen_num=config.DATA.MAX_GEN_NUM,
                debug_steps=config.REPORT_FREQ)
            logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                        f"Validation FID: {fid_score:.4f}, " +
                        f"time: {val_time:.2f}")
        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(
                config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
            paddle.save({"gen_state_dict":gen.state_dict(),
                         "dis_state_dict":dis.state_dict()}, model_path + '.pdparams')
            paddle.save({"gen_state_dict":gen_optimizer.state_dict(),
                         "dis_state_dict":dis_optimizer.state_dict()}, model_path + '.pdopt')
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")

if __name__ == "__main__":
    main()
