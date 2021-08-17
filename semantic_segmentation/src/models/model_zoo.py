import logging
import paddle

from collections import OrderedDict
from ..utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model(config):
    """
    Built the whole model, defined by `config.MODEL.META_ARCHITECTURE`.
    """
    model_name = config.MODEL.NAME
    model = MODEL_REGISTRY.get(model_name)(config)
    load_model_pretrain(model, config)
    return model


def load_model_pretrain(model, config):
    if config.PHASE == 'train':
        if config.MODEL.PRETRAINED:
            logging.info('load pretrained model from {}'.format(config.MODEL.PRETRAINED))
            state_dict_to_load = paddle.load(config.MODEL.PRETRAINED)
            keys_wrong_shape = []
            state_dict_suitable = OrderedDict()
            state_dict = model.state_dict()
            for k, v in state_dict_to_load.items():
                if v.shape == state_dict[k].shape:
                    state_dict_suitable[k] = v
                else:
                    keys_wrong_shape.append(k)
            logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
            msg = model.set_state_dict(state_dict_suitable)
            logging.info(msg)
    else:
        if config.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(config.TEST.TEST_MODEL_PATH))
            model_dic = paddle.load(config.TEST.TEST_MODEL_PATH)
            if 'state_dict' in model_dic.keys():
                # load the last checkpoint
                model_dic = model_dic['state_dict']
            msg = model.set_state_dict(model_dic)
            logging.info(msg)
