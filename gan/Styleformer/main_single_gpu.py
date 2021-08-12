
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

"""ViT training/validation using single GPU """

import sys
import os
import time
import logging
import argparse
import numpy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from datasets import get_dataloader
from datasets import get_dataset
from generator import Generator
# from discriminator_new import Discriminator
from discriminator import StyleGANv2Discriminator
from utils.utils import AverageMeter
from utils.utils import WarmupCosineScheduler
from config import get_config
from config import update_config
from metrics.fid import *
from PIL import Image

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
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')


# GP
def gradient_penalty(discriminator, real, fake, batchsize):
    # BATCH_SIZE,C,H,W = real.shape
    # 该OP返回数值服从范围[min, max)内均匀分布的随机Tensor，形状为 shape，数据类型为 dtype。
    #  epsilon ∼ U[0, 1].
    epsilon = paddle.randn((real.shape[0],1,1,1)).cuda()
    # 将epsilon扩展到real形状大小
    # x_hat = real * epsilon + fake * (1 - epsilon) 插值后的图片
    interpolated_images = paddle.to_tensor((real * epsilon + fake * (1 - epsilon)),stop_gradient=False)

    # 插值后的图片计算判别器得分
    mixed_scores = discriminator(interpolated_images)
    # print(mixed_scores)
    fake=paddle.to_tensor(paddle.ones((real.shape[0], 1)),stop_gradient=True).cuda()
    # 计算关于插值图的混合梯度
    # paddle. grad ( outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False, no_grad_vars=None )
    # 对于每个 inputs ，计算所有 outputs 相对于其的梯度和
    gradient = paddle.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # 做渐变点试图，把gradient展平
    gradient = paddle.reshape(gradient,(gradient.shape[0],-1))
    # L2 norm,计算范数
    # paddle.norm:该OP将计算给定Tensor的矩阵范数（Frobenius 范数）和向量范数（向量1范数、2范数、或者通常的p范数）.
    gradient_norm = gradient.norm(2,axis=1)
    # 计算gradient_penalty
    gradient_penalty = paddle.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def train(dataloader,
          gen,
          dis,
          gen_optimizer,
          dis_optimizer,
          epoch,
          total_batch,
          debug_steps=100,
          accum_iter=1):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """
    gen.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    time_st = time.time()
    LAMBDA_GP = 10
    # fid = FID()
    for batch_id, data in enumerate(dataloader):
        dis_optimizer.clear_grad()
        real_img = data[0]
        batch_size = real_img.shape[0]
        label = data[1]

        noise = paddle.randn([batch_size, 512])
        fake_img = gen(noise, c=paddle.zeros([0]))
        fake_img = (fake_img * 127.5 + 128).clip(0,255).astype('uint8')
        fake_img = fake_img / 255.0
        fake_pred = dis(fake_img.detach())
        real_pred = dis(real_img)

        # fid.update(fake_img, real_img)
        # fid_score = fid.accumulate()
        # print(fake_pred[0],real_pred[0])
        gp = gradient_penalty(dis, real_img, fake_img.detach(), batch_size)
        d_loss =  (
                -(paddle.mean(real_pred) - paddle.mean(fake_pred)) + LAMBDA_GP * gp
        )

        d_loss.backward()
        dis_optimizer.step()

        for _ in range(5):
            gen_optimizer.clear_grad()
            noise = paddle.randn([batch_size, 512])
            # print(noise[0],noise[1])
            gen_img = gen(noise, c=paddle.zeros([0]))
            # print(gen_img[0],gen_img[1])
            gen_img = (gen_img * 127.5 + 128).clip(0,255).astype('uint8')
            gen_img = gen_img / 255.0
            gen_imgs=paddle.multiply(gen_img,paddle.to_tensor(127.5))
            gen_imgs=paddle.clip(paddle.add(gen_imgs,paddle.to_tensor(127.5)).transpose((0,2,3,1)),min=0.0,max=255.0).astype('uint8')

            fake_pred = dis(gen_img)
            g_loss = -paddle.mean(fake_pred)

            g_loss.backward()
            gen_optimizer.step()

        gen_imgs = gen_imgs.numpy()

        for i in range(len(gen_imgs)):
            im = Image.fromarray(gen_imgs[i], 'RGB')
            im.save("./image/"+str(i)+".png")

        train_loss_meter.update(d_loss.numpy()[0] + g_loss.numpy()[0], batch_size)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"D Loss: {d_loss.item():.4f}, " + 
                f"G Loss: {g_loss.item():.4f},")
                # f"FID: {fid_score:.4f}"

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

def train_2(dataloader,
          gen,
          dis,
          gen_optimizer,
          dis_optimizer,
          epoch,
          total_batch,
          debug_steps=100,
          accum_iter=1):
    """Training for one epoch
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        epoch: int, current epoch
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
        accum_iter: int, num of iters for accumulating gradients
    Returns:
        train_loss_meter.avg
        train_acc_meter.avg
        train_time
    """
    gen.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    time_st = time.time()
    LAMBDA_GP = 10
    # fid = FID()
    l_d_total = 0
    for batch_id, data in enumerate(dataloader):
        dis_optimizer.clear_grad()
        real_img = data[0]

        batch_size = real_img.shape[0]

        noise = paddle.randn([batch_size, 512])
        fake_img = gen(noise, paddle.randint(0, 10, [batch_size]))
        fake_pred = dis(fake_img.detach())
        real_pred = dis(real_img)
        l_d_r1 = r1_penalty(real_pred, real_img)
        l_d_r1 = 10 * l_d_r1
        l_d_total += ld_r1
        l_d_total.backward()
        dis_optimizer.step()

        for _ in range(5):
            gen_optimizer.clear_grad()
            noise = paddle.randn([batch_size, 512])
        
            gen_img = gen(noise, paddle.randint(0, 10, [batch_size]))
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
                f"G Loss: {g_loss.item():.4f},")
                # f"FID: {fid_score:.4f}"

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_time


def validate(dataloader, model, criterion, total_batch, debug_steps=32):
    """Validation for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        criterion: nn.criterion
        total_epoch: int, total num of epoch, for logging
        debug_steps: int, num of iters to log info
    Returns:
        val_loss_meter.avg
        val_acc_meter.avg
        val_time
    """
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()
    time_st = time.time()
    fid = FID(32)
    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            curr_batch_size = data[0].shape[0]
            real_image = data[0]
            z = paddle.randn([curr_batch_size, 512])
            fake_image = model(z, c=paddle.randint(0, 10, [curr_batch_size]))

            fake_image = (fake_image * 127.5 + 128).clip(0,255).astype('uint8')
            fake_image = fake_image / 255.0
            fid.update(fake_image, real_image)


            if batch_id % debug_steps == 0:
                logger.info(
                    f"Val Step[{batch_id:04d}/{total_batch:04d}] done")
    fid_score = fid.accumulate()
    val_time = time.time() - time_st
    return fid_score, val_time


def main():
    # 0. Preparation
    last_epoch = config.TRAIN.LAST_EPOCH
    #paddle.set_device('gpu:0')
    # 1. Create model
    gen = Generator(config)
    # dis = Discriminator(c_dim=0,img_resolution=32,img_channels=3)
    dis = StyleGANv2Discriminator(config)
    #model = paddle.DataParallel(model)
    # 2. Create train and val dataloader
    dataset_train = get_dataset(config, mode='train')
    dataset_val = get_dataset(config, mode='val')
    dataloader_train = get_dataloader(config, dataset_train, 'train', False)
    dataloader_val = get_dataloader(config, dataset_val, 'val', False)
    # 3. Define criterion
    criterion = nn.CrossEntropyLoss()
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
        gen.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME and os.path.isfile(
            config.MODEL.RESUME+'.pdparams') and os.path.isfile(
                config.MODEL.RESUME+'.pdopt'):
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams')
        gen.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt')
        optimizer.set_state_dict(opt_state)
        logger.info(
            "----- Resume: Load model and optmizer from {config.MODEL.RESUME}")
    # 7. Validation
    if config.EVAL:
        logger.info('----- Start Validating')
        fid_score, val_time = validate(
            dataloader=dataloader_val,
            model=gen,
            criterion=criterion,
            total_batch=len(dataloader_val),
            debug_steps=config.REPORT_FREQ)
        logger.info(f"FID: {fid_score:.4f}, " +
                    f"time: {val_time:.2f}")
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
                                                  debug_steps=config.REPORT_FREQ,
                                                  accum_iter=config.TRAIN.ACCUM_ITER,
                                                  )
        scheduler.step()
        logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                    f"Train Loss: {train_loss:.4f}, " +
                    f"time: {train_time:.2f}")
        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            logger.info(f'----- Validation after Epoch: {epoch}')
            val_loss, val_acc1, val_acc5, val_time = validate(
                dataloader=dataloader_val,
                model=model,
                criterion=criterion,
                total_batch=len(dataloader_val),
                debug_steps=config.REPORT_FREQ)
            logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Acc@1: {val_acc1:.4f}, " +
                        f"Validation Acc@5: {val_acc5:.4f}, " +
                        f"time: {val_time:.2f}")
        # model save
        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            model_path = os.path.join(
                config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
            paddle.save(gen.state_dict(), model_path + '.pdparams')
            paddle.save(gen_optimizer.state_dict(), model_path + ".pdopt")
            logger.info(f"----- Save model: {model_path}.pdparams")
            logger.info(f"----- Save optim: {model_path}.pdopt")


if __name__ == "__main__":
    main()
