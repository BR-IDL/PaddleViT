import os
import paddle
import logging

from ...utils.download import download_file_and_uncompress
from ...utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, i.e. resnet.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""

model_urls = {
    'resnet18': 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
    'resnet34': 'https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
    'resnet50': 'https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
    'resnet101': 'https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
    'resnet152': 'https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
    'resnet50c': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/resnet50c.pdparams',
    'resnet101c': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/resnet101c.pdparams',
    'resnet152c': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/resnet152c.pdparams',
    'xception65': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/xception65.pdparams',
    'hrnet_w18_small_v1': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/hrnet_w18_small_v1.pdparams',
    'mobilenet_v2': 'https://github.com/GuoQuanhao/Trans2Seg-Paddle/releases/download/1/mobilenet_v2.pdparams',
}


def load_backbone_pretrained(model, backbone, config):
    if config.PHASE == 'train' and config.TRAIN.BACKBONE_PRETRAINED and (not config.MODEL.PRETRAINED):
        if os.path.isfile(config.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                config.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.set_state_dict(paddle.load(config.TRAIN.BACKBONE_PRETRAINED_PATH))
            logging.info('<All keys matched successfully>')
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            try:
                state_dict_path = paddle.utils.download.get_weights_path_from_url(model_urls[backbone], md5sum=None)
                model.set_state_dict(paddle.load(state_dict_path))
                msg = '<All keys matched successfully>'
            except Exception as e:
                logging.warning(e)
                logging.info('Use paddle download failed, try custom method!')
                model.set_state_dict(paddle.load(download_file_and_uncompress(model_urls[backbone])))
                msg = '<All keys matched successfully>'
            logging.info(msg)


def get_segmentation_backbone(backbone, config, norm_layer=paddle.nn.BatchNorm2D):
    """
    Built the backbone model, defined by `config.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(config, norm_layer)
    load_backbone_pretrained(model, backbone, config)
    return model
