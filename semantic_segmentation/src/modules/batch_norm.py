# this code heavily based on detectron2
import logging
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.autograd import PyLayer
from ..utils.distributed import get_world_size


class FrozenBatchNorm2d(nn.Layer):
    """
    BatchNorm2D where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", paddle.ones([num_features]))
        self.register_buffer("bias", paddle.zeros([num_features]))
        self.register_buffer("running_mean", paddle.zeros([num_features]))
        self.register_buffer("running_var", paddle.ones([num_features]) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = paddle.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = paddle.ones_like(self.running_var)

        if version is not None and version < 3:
            # logger = logging.getLogger(__name__)
            logging.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (paddle.nn.Layer):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2D, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data + module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def groupNorm(num_channels, eps=1e-5, momentum=0.1, affine=True):
    return nn.GroupNorm(min(32, num_channels), num_channels, eps=eps, affine=affine)


def get_norm(norm):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Layer or None: the normalization layer
    """
    support_norm_type = ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
    assert norm in support_norm_type, 'Unknown norm type {}, support norm types are {}'.format(
                                                                        norm, support_norm_type)
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2D,
            "SyncBN": NaiveSyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": groupNorm,
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm


class AllReduce(PyLayer):
    @staticmethod
    def forward(ctx, input):
        input_list = [paddle.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = paddle.stack(input_list, axis=0)
        return paddle.sum(inputs, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm(nn.BatchNorm2D):
    """
    `paddle.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = paddle.mean(input, axis=[0, 2, 3])
        meansqr = paddle.mean(input * input, axis=[0, 2, 3])

        vec = paddle.concat([mean, meansqr], axis=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = paddle.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = paddle.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
