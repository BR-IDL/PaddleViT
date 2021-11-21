from paddle import optimizer as optim


def get_optimizer(model, lr_scheduler, config):
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters=model.parameters(), learning_rate=lr_scheduler, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=lr_scheduler, epsilon=config.TRAIN.EPSILON, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters=model.parameters(), learning_rate=lr_scheduler, epsilon=config.TRAIN.EPSILON, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(
            parameters=model.parameters(), learning_rate=lr_scheduler, epsilon=config.TRAIN.EPSILON, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(
           parameters= model.parameters(), learning_rate=lr_scheduler, alpha=0.9, epsilon=config.TRAIN.EPSILON,
            momentum=config.TRAIN.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Expected optimizer method in [sgd, adam, adadelta, rmsprop], but received "
                         "{}".format(opt_lower))

    return optimizer
