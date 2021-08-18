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

"""mixup and cutmix for batch data"""
import numpy as np
import paddle


def rand_bbox(image_shape, lam, count=None):
    """ CutMix bbox by lam value
    Generate 1 random bbox by value lam. lam is the cut size rate.
    The cut_size is computed by sqrt(1-lam) * image_size.

    Args:
        image_shape: tuple/list, image height and width
        lam: float, cutmix lambda value
        count: int, number of bbox to generate
    """
    image_h, image_w = image_shape[-2:]
    cut_rate = np.sqrt(1. - lam)
    cut_h = int(cut_rate * image_h)
    cut_w = int(cut_rate * image_w)

    # get random bbox center
    cy = np.random.randint(0, image_h, size=count)
    cx = np.random.randint(0, image_w, size=count)

    # get bbox coords
    bbox_x1 = np.clip(cx - cut_w // 2, 0, image_w)
    bbox_y1 = np.clip(cy - cut_h // 2, 0, image_h)
    bbox_x2 = np.clip(cx + cut_w // 2, 0, image_w)
    bbox_y2 = np.clip(cy + cut_h // 2, 0, image_h)

    # NOTE: in paddle, tensor indexing e.g., a[x1:x2],
    # if x1 == x2, paddle will raise ValueErros, 
    # while in pytorch, it will return [] tensor
    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


def rand_bbox_minmax(image_shape, minmax, count=None):
    """ CutMix bbox by min and max value
    Generate 1 random bbox by min and max percentage values.
    Minmax is a tuple/list of min and max percentage vlaues
    applied to the image width and height.

    Args:
        image_shape: tuple/list, image height and width
        minmax: tuple/list, min and max percentage values of image size
        count: int, number of bbox to generate
    """
    assert len(minmax) == 2
    image_h, image_w = image_shape[-2:]
    min_ratio = minmax[0]
    max_ratio = minmax[1]
    cut_h = np.random.randint(int(image_h * min_ratio), int(image_h * max_ratio), size=count) 
    cut_w = np.random.randint(int(image_w * min_ratio), int(image_w * max_ratio), size=count) 

    bbox_x1 = np.random.randint(0, image_w - cut_w, size=count)
    bbox_y1 = np.random.randint(0, image_h - cut_h, size=count)
    bbox_x2 = bbox_x1 + cut_w
    bbox_y2 = bbox_y1 + cut_h

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2


def cutmix_generate_bbox_adjust_lam(image_shape, lam, minmax=None, correct_lam=True, count=None):
    """Generate bbox and apply correction for lambda
    If the mimmax is None, apply the standard cutmix by lam value,
    If the minmax is set, apply the cutmix by min and max percentage values.

    Args:
        image_shape: tuple/list, image height and width
        lam: float, cutmix lambda value
        minmax: tuple/list, min and max percentage values of image size
        correct_lam: bool, if True, correct the lam value by the generated bbox
        count: int, number of bbox to generate
    """
    if minmax is not None:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = rand_bbox_minmax(image_shape, minmax, count)
    else:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = rand_bbox(image_shape, lam, count)

    if correct_lam or minmax is not None:
        image_h, image_w = image_shape[-2:]
        bbox_area = (bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1)
        lam = 1. - bbox_area / float(image_h * image_w)
    return (bbox_x1, bbox_y1, bbox_x2, bbox_y2), lam


def one_hot(x, num_classes, on_value=1., off_value=0.):
    """ Generate one-hot vector for label smoothing
    Args:
        x: tensor, contains label/class indices
        num_classes: int, num of classes (len of the one-hot vector)
        on_value: float, the vector value at label index, default=1.
        off_value: float, the vector value at non-label indices, default=0.
    Returns:
        one_hot: tensor, tensor with on value at label index and off value
                 at non-label indices.
    """
    x = x.reshape_([-1, 1])
    x_smoothed = paddle.full((x.shape[0], num_classes), fill_value=off_value)
    for i in range(x.shape[0]):
        x_smoothed[i, x[i]] = on_value
    return x_smoothed


def mixup_one_hot(label, num_classes, lam=1., smoothing=0.):
    """ mixup and label smoothing in batch
    label smoothing is firstly applied, then
    mixup is applied by mixing the bacth and its flip,
    with a mixup rate.

    Args:
        label: tensor, label tensor with shape [N], contains the class indices
        num_classes: int, num of all classes
        lam: float, mixup rate, default=1.0
        smoothing: float, label smoothing rate
    """
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(label, num_classes, on_value, off_value)
    y2 = one_hot(label.flip(axis=[0]), num_classes, on_value, off_value)
    return y2 * (1 - lam) + y1 * lam


class Mixup:
    """Mixup class
    Args:
        mixup_alpha: float, mixup alpha for beta distribution, default=1.0,
        cutmix_alpha: float, cutmix alpha for beta distribution, default=0.0,
        cutmix_minmax: list/tuple, min and max value for cutmix ratio, default=None,
        prob: float, if random prob < prob, do not use mixup, default=1.0,
        switch_prob: float, prob of switching mixup and cutmix, default=0.5,
        mode: string, mixup up, now only 'batch' is supported, default='batch',
        correct_lam: bool, if True, apply correction of lam, default=True,
        label_smoothing: float, label smoothing rate, default=0.1,
        num_classes: int, num of classes, default=1000
    """
    def __init__(self,
                 mixup_alpha=1.0,
                 cutmix_alpha=0.0,
                 cutmix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 correct_lam=True,
                 label_smoothing=0.1,
                 num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if cutmix_minmax is not None:
            assert len(cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        assert mode == 'batch', 'Now only batch mode is supported!'

    def __call__(self, x, target):
        assert x.shape[0] % 2 == 0, "Batch size should be even"
        lam = self._mix_batch(x)
        target = mixup_one_hot(target, self.num_classes, lam, self.label_smoothing)
        return x, target

    def get_params(self):
        """Decide to use cutmix or regular mixup by sampling and
           sample lambda for mixup
        """
        lam = 1.
        use_cutmix = False
        use_mixup = np.random.rand() < self.mix_prob
        if use_mixup:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
                lam_mix = np.random.beta(alpha, alpha)
            elif self.mixup_alpha == 0. and self.cutmix_alpha > 0.:
                use_cutmix=True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            elif self.mixup_alpha > 0. and self.cutmix_alpha == 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            else:
                raise ValueError('mixup_alpha and cutmix_alpha cannot be all 0')
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        """mixup/cutmix by adding batch data and its flipped version"""
        lam, use_cutmix = self.get_params()
        if lam == 1.:
            return lam
        if use_cutmix:
            (bbox_x1, bbox_y1, bbox_x2, bbox_y2), lam = cutmix_generate_bbox_adjust_lam(
                x.shape,
                lam,
                minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam)

            # NOTE: in paddle, tensor indexing e.g., a[x1:x2],
            # if x1 == x2, paddle will raise ValueErros, 
            # but in pytorch, it will return [] tensor without errors
            if int(bbox_x1) != int(bbox_x2) and int(bbox_y1) != int(bbox_y2):
                x[:, :, int(bbox_x1): int(bbox_x2), int(bbox_y1): int(bbox_y2)] = x.flip(axis=[0])[
                    :, :, int(bbox_x1): int(bbox_x2), int(bbox_y1): int(bbox_y2)]
        else:
            x_flipped = x.flip(axis=[0])
            x_flipped = x_flipped * (1 - lam)
            x.set_value(x * (lam) + x_flipped)
        return lam
