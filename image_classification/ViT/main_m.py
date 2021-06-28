import sys
import os
import time
import logging
import configparser
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from datasets import get_loader, get_dataset
from models.transformer import *
from utils.utils import *
from config import *


log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m%d %I:%M:%S %p")

config = get_config()

if not config.EVAL:
    config.SAVE = '{}/train-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))
else:
    config.SAVE = '{}/eval-{}'.format(config.SAVE, time.strftime('%Y%m%d-%H-%M-%S'))

if not os.path.exists(config.SAVE):
    os.makedirs(config.SAVE, exist_ok=True)

logger = logging.getLogger()
fh = logging.FileHandler(os.path.join(config.SAVE, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)
logger.info(f'config= {config}')


def train(dataloader, model, criterion, optimizer, epoch, total_batch, scheduler=None, debug_steps=100):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    time_st = time.time()

    #print(f'----- len dataloader: {len(dataloader)}')
    for batch_id, data in enumerate(dataloader):
        #print(f'----- batch_size(in loop): {data[0].shape[0]}')
        image = data[0]
        label = data[1]

        output, _  = model(image)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()
        
        pred = F.softmax(output)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(1))

        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        train_acc_meter.update(acc.numpy()[0], n)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " + 
                f"Step[{batch_id:04d}/{total_batch:04d}], " + 
                f"Avg Loss: {train_loss_meter.avg:.4f}, " + 
                f"Avg Acc: {train_acc_meter.avg:.4f}")

    train_time = time.time() - time_st
    return train_loss_meter.avg, train_acc_meter.avg, train_time


def validate(dataloader, model, criterion, epoch, total_batch, debug_steps=100):
    model.eval()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    time_st = time.time()

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        output, _ = model(image)
        loss = criterion(output, label)

        pred = F.softmax(output)
        acc = paddle.metric.accuracy(pred, label.unsqueeze(1))

        n = image.shape[0]
        val_loss_meter.update(loss.numpy()[0], n)
        val_acc_meter.update(acc.numpy()[0], n)

        if batch_id % debug_steps == 0:
            logger.info(
                f"Val Step[{batch_id:04d}/{total_batch:04d}], " + 
                f"Avg Loss: {val_loss_meter.avg:.4f}, " + 
                f"Avg Acc: {val_acc_meter.avg:.4f}")

    train_time = time.time() - time_st
    return val_loss_meter.avg, val_acc_meter.avg, val_time


def main_worker(dataset_train, dataset_val):
    # 0. Preparation
    last_epoch = 0
    dist.init_parallel_env()
    #paddle.set_device('gpu:0')
    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    # 1. Create model
    model = VisualTransformer(config)
    model = paddle.DataParallel(model)
    # 2. Create train and val dataloader
    dataloader_train, dataloader_val = get_loader(config=config.DATA, 
                                                  dataset_train=dataset_train,
                                                  dataset_test=dataset_val,
                                                  multi=True)
    total_batch_train = len(dataloader_train)//world_size
    total_batch_val = len(dataloader_val)//world_size
    #print(total_batch_train)
    #print(total_batch_val)

    # 3. Define criterion
    criterion = nn.CrossEntropyLoss()
    # 4. Define optimizer and lr_scheduler
    scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == "cosine":
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

    if config.TRAIN.OPTIMIZER.NAME == "SGD":
        optimizer = paddle.optimizer.SGD(learning_rate= scheduler if scheduler is not None else config.TRAIN.BASE_LR,
                                     weight_decay=config.TRAIN.WEIGHT_DECAY,
                                     grad_clip=config.TRAIN.GRAD_CLIP,
                                     epsilon=config.TRAIN.OPTIMIZER.EPS,
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

    # 5. Load pretrained model / load resumt model and optimizer states
    if config.MODEL.PRETRAINED and os.path.isfile( config.MODEL.PRETRAINED + '.pdparams'):
        model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams') 
        model.set_dict(model_state)
        logger.info(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    if config.MODEL.RESUME and os.path.isfile(config.MODEL.RESUME+'.pdparams') and os.path.isfile(config.MODEL.RESUME+'.pdopt'):
        model_state = paddle.load(config.MODEL.RESUME+'.pdparams') 
        model.set_dict(model_state)
        opt_state = paddle.load(config.MODEL.RESUME+'.pdopt') 
        optimizer.set_dict(opt_state)
        logger.info(f"----- Resume Training: Load model and optmizer states from {config.MODEL.RESUME}")

    # 6. Start training and validation
    if local_rank == 0:
        logging.info(f"Start training from epoch {last_epoch+1}.")
    for epoch in range(last_epoch+1, config.TRAIN.NUM_EPOCHS+1):
        # train
        train_loss, train_acc, train_time = train(dataloader=dataloader_train,
                                                  model=model, 
                                                  criterion=criterion, 
                                                  optimizer=optimizer, 
                                                  epoch=epoch,
                                                  total_batch=total_batch_train,
                                                  scheduler=scheduler, 
                                                  debug_steps=config.REPORT_FREQ,
                                                  )
        if local_rank == 0:
            logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                        f"Train Loss: {train_loss:.4f}, " +
                        f"Train Acc: {train_acc:.4f}, " +
                        f"time: {train_time:.2f}")
        # validation
        if epoch % config.VALIDATE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
            if local_rank == 0:
                logger.info(f'----- Validation after Epoch: {epoch}')
            val_loss, val_acc, val_time = test(dataloader=dataloader_val,
                                               model=model,
                                               criterion=criterion,
                                               epoch=epoch,
                                               total_batch=total_batch_val,
                                               debug_steps=config.REPORT_FREQ)
            logger.info(f"----- Epoch[{epoch:03d}/{config.TRAIN.NUM_EPOCHS:03d}], " +
                        f"Validation Loss: {val_loss:.4f}, " +
                        f"Validation Acc: {val_acc:.4f}, " +
                        f"time: {val_time:.2f}")
        # model save
        if local_rank == 0:
            if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.NUM_EPOCHS:
                model_path = os.path.join(config.SAVE, f"{config.MODEL.TYPE}-Epoch-{epoch}-Loss-{train_loss}")
                paddle.save(model.state_dict(), model_path)
                paddle.save(optimizer.state_dict(), model_path)
                logger.info(f"----- Save model: {model_path}.pdparams")
                logger.info(f"----- Save optim: {model_path}.pdopt")


def main():
    dataset_train, dataset_val = get_dataset(config.DATA)
    dist.spawn(main_worker, args=(dataset_train, dataset_val, ), nprocs=paddle.distributed.get_world_size())

if __name__ == "__main__":
    main()
