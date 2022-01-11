import paddle
import paddle.nn as nn

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1,)*(x.ndim-1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    #random_tensor.to(x.device)
    random_tensor = random_tensor.floor()
    output = x.divide(keep_prob) * random_tensor
    return output
    

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
        
