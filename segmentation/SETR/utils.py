import json
import paddle
import matplotlib.pyplot as plt
import math
import re
import numpy as np
from paddle.optimizer.lr import LRScheduler
from config import *


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



def get_exclude_from_weight_decay_fn(exclude_list=[]):
    if len(exclude_list) == 0:
        exclude_from_weight_decay_fn = None
    else:
        def exclude_fn(param):
            for name in exclude_list:
                if param.endswith(name):
                    return True
            return False
        exclude_from_weight_decay_fn = exclude_fn
    return exclude_from_weight_decay_fn


class WarmupCosineScheduler(LRScheduler):
    """
    Linearly increase learning rate from "warmup_start_lr" to "start_lr" over "warmup_epochs"
    Cosinely decrease learning rate from "start_lr" to "end_lr" over remaining "total_epochs - warmup_epochs".

    learning_rate: the starting learning rate (without warmup)
    warmup_start_lr: warmup starting learning rate 
    start_lr: the starting learning rate (without warmup)
    end_lr: the ending learning rate after whole loop
    warmup_epochs: # of epochs for warmup
    total_epochs: # of total epochs (include warmup)

    """
    def __init__(self, learning_rate, warmup_start_lr, start_lr, end_lr, warmup_epochs, total_epochs, cycles=0.5, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.cycles = cycles
        super(WarmupCosineScheduler, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            val = (self.start_lr - self.warmup_start_lr) * float(
                        self.last_epoch)/float(self.warmup_epochs) + self.warmup_start_lr
            return val
        else:
            progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            val = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
            val = max(0.0, val * (self.start_lr - self.end_lr) + self.end_lr)
            return val



class DictObj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


#def dict2obj(dict1):
#    return json.loads(json.dumps(dict1), object_hook=DictObj)
#
#
#def obj2dict(obj):
#    out = dict()
#    d = obj.__dict__
#    for key, val in d.items():
#        if isinstance(val, DictObj):
#            out[key] = obj2dict(val)
#        else:
#            out[key] = val
#    return out
#
#def config2dict(config):
#    res = {}
#    for section in config.sections():
#        res[section] = {}
#        for key in config[section]:
#            val = config[section][key]
#            if isfloat(val):
#                val = float(val)
#            elif val.isnumeric():
#                val = int(val)
#            elif val in ["True", "False", "true", "false"]:
#                val = eval(val)
#            elif val == "None":
#                val = None
#            res[section][key] = val
#    return res
#
#def update_config(config, args):
#    args_dict = vars(args)
#    for section in config.sections():
#        for key in config[section]:
#            if key in args_dict.keys() and args_dict[key] is not None:
#                config[section][key] = str(args_dict[key])
#    return config
#
#
## check str float
#def isfloat(s):
#    if re.match(r'^-?\d+(?:\.\d+)$', s) is None:
#        return False
#    else:
#        return True




def main():

    config = get_config()

    #sch =  WarmupCosineScheduler(
    #        learning_rate=0.1,
    #        warmup_start_lr = 0.0,
    #        start_lr = 0.4,
    #        end_lr = 0.0,
    #        warmup_epochs=10,
    #        total_epochs=120,
    #        )
    sch = WarmupCosineScheduler(learning_rate=config.TRAIN.BASE_LR,
                                       warmup_start_lr=config.TRAIN.WARMUP_START_LR,
                                       start_lr=config.TRAIN.BASE_LR,
                                       end_lr=config.TRAIN.END_LR,
                                       warmup_epochs=config.TRAIN.WARMUP_EPOCHS,
                                       total_epochs=config.TRAIN.NUM_EPOCHS,
                                       last_epoch=config.TRAIN.LAST_EPOCH,)

    linear = paddle.nn.Linear(10, 10)

    sgd = paddle.optimizer.SGD(learning_rate=sch, parameters=linear.parameters())

    lr = [[0,0]]
    for i in range(config.TRAIN.NUM_EPOCHS):
        #sch.step()
        #lr.append([i+1,sch.state_dict()['last_lr']])
        #print(sch.state_dict()['last_lr'])

        print(sgd.get_lr())
        sch.step()
        lr.append([i+1, sgd.get_lr()])

    lr = np.array(lr)

    
    plt.plot(lr[:,0], lr[:,1])
    plt.savefig('lr.png')

    


if __name__ == "__main__":
    main()
