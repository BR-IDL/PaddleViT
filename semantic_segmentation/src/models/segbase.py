"""Base Model for Semantic Segmentation"""
import math
import numbers
import paddle.nn as nn
import paddle.nn.functional as F

from .backbones import get_segmentation_backbone
from ..modules import get_norm

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Layer):
    r"""Base Model for Semantic Segmentation
    """
    def __init__(self, config, need_backbone=True):
        super(SegBaseModel, self).__init__()
        self.nclass = config.DATA.NUM_CLASSES
        self.aux = config.TRAIN.LR_SCHEDULER.AUX
        self.norm_layer = get_norm(config.MODEL.BN_TYPE)
        self.backbone = None
        self.encoder = None
        if need_backbone:
            self.get_backbone(config)

    def get_backbone(self, config):
        self.backbone = config.MODEL.BACKBONE.lower()
        self.encoder = get_segmentation_backbone(self.backbone, config, self.norm_layer)

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.encoder(x)
        return c1, c2, c3, c4

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

    def evaluate(self, image, config):
        """evaluating network with inputs and targets"""
        scales = config.VAL.SCALE_RATIOS
        flip = config.VAL.FLIP
        crop_size = _to_tuple(config.VAL.CROP_SIZE) if config.VAL.CROP_SIZE else None
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = paddle.zeros((batch, self.nclass, h, w)).cuda()
        scores = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            if crop_size is not None:
                assert crop_size[0] >= h and crop_size[1] >= w
                crop_size_scaled = (int(math.ceil(crop_size[0] * scale)),
                                    int(math.ceil(crop_size[1] * scale)))
                cur_img = _pad_image(cur_img, crop_size_scaled)
            outputs = self.forward(cur_img)[0][..., :height, :width]
            if flip:
                outputs += _flip_image(self.forward(_flip_image(cur_img))[0])[..., :height, :width]

            score = _resize_image(outputs, h, w)

            if scores is None:
                scores = score
            else:
                scores += score
        return scores


def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


def _pad_image(img, crop_size):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size[0] - h if h < crop_size[0] else 0
    padw = crop_size[1] - w if w < crop_size[1] else 0
    if padh == 0 and padw == 0:
        return img
    img_pad = F.pad(img, (0, padh, 0, padw))

    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip((3))


def _to_tuple(size):
    if isinstance(size, (list, tuple)):
        assert len(size), 'Expect eval crop size contains two element, ' \
                          'but received {}'.format(len(size))
        return tuple(size)
    elif isinstance(size, numbers.Number):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))
