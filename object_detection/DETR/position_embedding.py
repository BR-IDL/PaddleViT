import copy
import math
import paddle
import paddle.nn as nn


class PositionEmbeddingSine(nn.Layer):
    def __init__(self, num_position_features=64, temp=10000, norm=False, scale=None):
        super().__init__()
        self.num_position_features = num_position_features
        self.temp = temp
        self.norm = norm
        if scale is not None and norm is False:
            raise ValueError('norm should be true if scale is passed')
        self.scale = 2 * math.pi if scale is None else scale 

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = (mask < 0.5).astype('float32')
        y_embed = not_mask.cumsum(1, dtype='float32')
        x_embed = not_mask.cumsum(2, dtype='float32')
        if self.norm:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(self.num_position_features, dtype='int32')
        dim_t = self.temp ** (2 * (dim_t // 2) / self.num_position_features)

        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = x_embed.unsqueeze(-1) / dim_t

        pos_y = paddle.stack((pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_x = paddle.stack((pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])

        return pos


class PositionEmbeddingLearned(nn.Layer):
    def __init__(self, num_position_features=256):
        super().__init__()
        w_attr_1 = self.init_weights()
        self.row_embed = nn.Embedding(50, num_position_features, weight_attr=w_attr_1)
        w_attr_2 = self.init_weights()
        self.col_embed = nn.Embedding(50, num_position_features, weight_attr=w_attr_2)

    def init_weights(self):
        return paddle.ParamAttr(initializer=nn.initializer.Uniform(low=0., high=1.))

    def forward(self, tensor_list):
        x = tensor_list.tensors #[batch, 2048(R50 feature), H, W]
        h, w = x.shape[-2:]
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)

        pos = paddle.concat([
            x_embed.unsqueeze(0).expand((h, x_embed.shape[0], x_embed.shape[1])),
            y_embed.unsqueeze(1).expand((y_embed.shape[0], w, y_embed.shape[1])),
            ], axis=-1)
        pos = pos.transpose([2, 0, 1]) #[dim, h, w]
        pos = pos.unsqueeze(0) #[1, dim, h, w]
        pos = pos.expand([x.shape[0]] + pose.shape[1::]) #[batch, dim, h, w]
        return pos
        

def build_position_encoding(embed_dim=245, mode='sine'):
    """generate position encoding (sine) or position embedding (learned)"""
    assert mode in ['sine', 'learned']
    N_steps = embed_dim // 2
    if mode == 'sine':
        position_embedding = PositionEmbeddingSine(N_steps, norm=True)
    else: # learned
        position_embedding = PositionEmbeddingLearned(N_steps)
    return position_embedding
