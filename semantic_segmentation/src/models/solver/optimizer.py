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

"""
Create Optimizer
"""
from paddle import optimizer as optim
from paddle.nn import ClipGradByGlobalNorm

def get_optimizer(model, lr_scheduler, config):
    """Get Optimizer for Training

    Attributes:
        model: nn.Layer, training model
        lr_scheduler: (LRScheduler|float), learning rate scheduler
        config: CfgNode, hyper for optimizer
    """
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    clip = None
    if config.TRAIN.OPTIMIZER.GRAD_CLIP:
        clip = ClipGradByGlobalNorm(config.TRAIN.OPTIMIZER.GRAD_CLIP)

    if opt_lower == 'sgd':
        optimizer = optim.Momentum(parameters=model.parameters(),
                                   learning_rate=lr_scheduler,
                                   momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                   use_nesterov=config.TRAIN.OPTIMIZER.NESTEROV,
                                   weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY),
                                   grad_clip=clip)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters=model.parameters(),
                               learning_rate=lr_scheduler,
                               epsilon=config.TRAIN.OPTIMIZER.EPS,
                               weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY))
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters=model.parameters(),
                                learning_rate=lr_scheduler,
                                beta1=config.TRAIN.OPTIMIZER.BETAS[0],
                                beta2=config.TRAIN.OPTIMIZER.BETAS[1],
                                epsilon=config.TRAIN.OPTIMIZER.EPS,
                                weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY),
                                grad_clip=clip)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters=model.parameters(),
                                   rho=config.TRAIN.OPTIMIZER.RHO,
                                   learning_rate=lr_scheduler,
                                   epsilon=config.TRAIN.OPTIMIZER.EPS,
                                   weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY),
                                   grad_clip=clip)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSProp(parameters=model.parameters(),
                                  rho=config.TRAIN.OPTIMIZER.RHO,
                                  momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                  learning_rate=lr_scheduler,
                                  centered=config.TRAIN.OPTIMIZER.CENTERTED,
                                  epsilon=config.TRAIN.OPTIMIZER.EPS,
                                  weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY),
                                  grad_clip=clip)
    else:
        raise ValueError("Expected optimizer method in [SGD, Adam, AdamW, Adadelta, RMSProp],"
                         "but received {}".format(opt_lower))
    return optimizer
