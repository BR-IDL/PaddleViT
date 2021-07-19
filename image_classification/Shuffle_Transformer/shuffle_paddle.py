'''
    This module implement a shuffle transformer model.
'''
import paddle
import paddle.nn as nn
import numpy as np
from droppath import DropPath



class MLP(nn.Layer):
    '''
    Describe:
        A Feed Forward Network, which use Conv replace the Linear.
    '''
    def __init__(self,
                 in_feature,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU6,
                 drop=0.,
                 stride=False):
        '''
        Args:
            in_feature:       The MLP's input feature dim.
            hidden_features:  The MLP's hidden feature dim.
            out_features:     The MLP's output feature dim.
            act_layer:        The act function of Conv2D.
            drop:             The drop rate of dropout.
            stride:           Emmmmmmm, what the fuck...
        '''
        super().__init__()
        self.stride = stride
        out_feature = out_features or in_feature
        hid_feature = hidden_features or in_feature
        self.fc1    = nn.Conv2D(in_feature, hid_feature, 1, 1, 0)
        self.act    = act_layer()
        self.fc2    = nn.Conv2D(hid_feature, out_feature, 1, 1, 0)
        self.drop   = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Layer):
    '''
    Describe:
        The Basic Multi-Window-Self-Attention modules.
    '''
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=1,
                 shuffle=False,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 relative_pos_embedding=False):
        '''
        Args:
            dim:                    The input dim of Attention modules.
            num_heads:              The head_number of Attention modules.
            windows_size:           The window size of Attention modules.
            shuffle:                Whether this modules will shuffle the output.
            qkv_bias:               Whether this modules's q,k,v need the bias.
            qk_scale:               Emmmmmm, what the fucking shit.
            attn_drop:              The drop rate of attention modules.
            proj_drop:              The drop rate of projection modules.
            relative_pos_embedding: Whether this modules nedd the position embedding.
        '''
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim       = dim // self.num_heads
        self.ws        = window_size
        self.shuffle   = shuffle
        self.scale     = qk_scale or head_dim ** -0.5

        self.to_qkv    = nn.Conv2D(dim, dim * 3, 1, bias_attr=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Conv2D(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = \
            paddle.create_parameter([(2 * window_size - 1) * (2 * window_size - 1), num_heads],
                                    dtype='float32',
                                    default_initializer=\
                                        paddle.nn.initializer.TruncatedNormal(std=.02))

            coords_h   = paddle.arange(0, self.ws)
            coords_w   = paddle.arange(0, self.ws)
            coords     = paddle.stack(paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(coords, 1) # [2, window_h * window_w]
            # 2, window_h * window_w, window_h * window_h
            relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)
            # winwod_h*window_w, window_h*window_w, 2
            relative_coords = relative_coords.transpose([1, 2, 0])
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2* self.ws - 1
            # [window_size * window_size, window_size*window_size]
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

    def reshap_and_transpose_front(self, tensor):
        '''
        Describe:
            This function used to replace the einops.
            rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d')
        '''
        b0, c0, h0, w0 = tensor.shape
        if self.shuffle:
            # reshape the tensor from
            # b,(qkv h d),(ws1 hh),(ws2 ww) --> b, qkv, h, d, ws1, hh, ws2, ww
            tensor         = paddle.reshape(tensor,
                                            shape=[b0,
                                                   3,
                                                   self.num_heads,
                                                   c0 // (3 * self.num_heads),
                                                   self.ws,
                                                   h0 // self.ws,
                                                   self.ws,
                                                   w0 // self.ws])
            b, qkv, h, d, ws1, hh, ws2, ww = tensor.shape
            # transpose the tensor to qkv,b,hh,ww,h,ws1,ws2,d
            tensor         = paddle.transpose(tensor, perm=[1, 0, 5, 7, 2, 4, 6, 3])
            # reshape the tensor to qkv,(b hh ww),h,(ws1 ws2),d
            tensor         = paddle.reshape(tensor, shape=[3, (b * hh * ww), h, (ws1 * ws2), d])
        else:
            # reshape the tensor from b,
            # (qkv h d),(ws1 hh),(ws2 ww) --> b, qkv, h, d, hh, ws1, ww, ws2
            tensor         = paddle.reshape(tensor,
                                            shape=[b0,
                                                   3,
                                                   self.num_heads,
                                                   c0 // (3 * self.num_heads),
                                                   h0 // self.ws,
                                                   self.ws,
                                                   w0 // self.ws,
                                                   self.ws])
            b, qkv, h, d, hh, ws1, ww, ws2 = tensor.shape
            # transpose the tensor to qkv,b,hh,ww,h,ws1,ws2,d
            tensor         = paddle.transpose(tensor, perm=[1, 0, 4, 6, 2, 5, 7, 3])
            # reshape the tensor to qkv,(b hh ww),h,(ws1 ws2),d
            tensor         = paddle.reshape(tensor, shape=[3, (b * hh * ww), h, (ws1 * ws2), d])
        q, k, v        = paddle.unbind(tensor, axis=0)
        return q, k, v

    def reshape_and_transpose_back(self, tensor, origin_shape):
        '''
        Describe:
            This function used to replace the einops.
            Rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)')
        '''
        b0, c0, h0, w0 = origin_shape
        if self.shuffle:
            # reshape the tensor to b, hh, ww, h, ws1, ws2, d
            tensor     = paddle.reshape(tensor,
                                        shape=[b0,
                                               h0 // self.ws,
                                               w0 // self.ws,
                                               self.num_heads,
                                               self.ws, self.ws,
                                               c0 // self.num_heads])
            b, hh, ww, h, ws1, ws2, d = tensor.shape
            # reshape the tensor to b, h, d, ws1, hh, ws2, ww
            tensor     = paddle.transpose(tensor, perm=[0, 3, 6, 4, 1, 5, 2])
            tensor     = paddle.reshape(tensor, shape=[b, (h * d), (ws1 * hh), (ws2 * ww)])
        else:
            # reshape the tensor to b, hh, ww, h, ws1, ws2, d
            tensor     = paddle.reshape(tensor,
                                        shape=[b0,
                                               h0 // self.ws,
                                               w0 // self.ws,
                                               self.num_heads,
                                               self.ws,
                                               self.ws,
                                               c0 // self.num_heads])
            b, hh, ww, h, ws1, ws2, d = tensor.shape
            # reshape the tensor to b, h, d, hh, ws1, ww, ws2
            tensor     = paddle.transpose(tensor, perm=[0, 3, 6, 1, 4, 2, 5])
            tensor     = paddle.reshape(tensor, shape=[b, (h * d), (hh * ws1), (ww * ws2)])
        return tensor


    def get_relative_pos_bias_from_pos_index(self):
        '''
        Describe:
            This function used to get relative_position_bias by position index and position table.
        '''
        # relative_position_bias_table is a ParamBase object
        table = self.relative_position_bias_table # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1])
        # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias


    def forward(self, x):
        origin_shape = x.shape
        qkv          = self.to_qkv(x)
        q, k, v      = self.reshap_and_transpose_front(qkv)

        dot          = paddle.matmul(q, paddle.transpose(k, perm=[0, 1, 3, 2])) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.get_relative_pos_bias_from_pos_index()
            relative_position_bias = paddle.reshape(relative_position_bias,
                                                    shape=[self.ws * self.ws,
                                                           self.ws * self.ws,
                                                           -1])
            relative_position_bias = paddle.transpose(relative_position_bias, perm=[2, 0, 1])
            dot      += paddle.unsqueeze(relative_position_bias, axis=0)

        attn         = self.softmax(dot)
        out          = paddle.matmul(attn, v)

        out          = self.reshape_and_transpose_back(out, origin_shape)

        out          = self.proj(out)
        out          = self.proj_drop(out)
        return out



class Block(nn.Layer):
    '''
    Describe:
        The bacical model of shuffle_transformer.
    '''
    def __init__(self,
                 dim,
                 out_dim,
                 num_heads,
                 window_size = 1,
                 shuffle=False,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2D,
                 stride=False,
                 relative_pos_embedding=False):
        '''
        Args:
            dim:             The input data's dimension.
            out_dim:         The output data's dimension.
            mlp_ratio:       The mlp ratio
        '''
        super().__init__()
        self.input_dim    = dim
        self.output_dim   = out_dim
        self.norm1        = norm_layer(dim, 0.1)
        self.attn         = Attention(dim,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      shuffle=shuffle,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      attn_drop=attn_drop,
                                      proj_drop=drop,
                                      relative_pos_embedding=relative_pos_embedding)

        self.local        = nn.Conv2D(dim, dim, window_size, 1, window_size // 2, groups=dim)
        self.drop_path    = DropPath(drop_path)
        self.norm2        = norm_layer(dim, 0.1)

        mlp_hidden_dim    = int(dim * mlp_ratio)
        self.mlp          = MLP(dim,
                                mlp_hidden_dim,
                                out_dim,
                                act_layer=act_layer,
                                drop=drop,
                                stride=stride)
        self.norm3        = norm_layer(out_dim, 0.1)



    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.local(self.norm2(x))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class PatchMerging(nn.Layer):
    '''
    Describe:
        Use this class to Merging different channel's window transformer output.
    '''
    def __init__(self, in_dim=32, out_dim=64, norm_layer=nn.BatchNorm2D):
        super().__init__()
        self.dim      = in_dim
        self.out_dim  = out_dim
        self.norm     = norm_layer(in_dim, 0.1)
        self.reduction= nn.Conv2D(in_dim, out_dim, 2, 2, 0, bias_attr=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

class StageModule(nn.Layer):
    '''
    Describe:
        This class is a Window Transformer's encapsulation.
    '''
    def __init__(self,
                 layers,
                 dim,
                 out_dim,
                 num_heads,
                 window_size=1,
                 shuffle=True,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2D,
                 relative_pos_embedding=False):
        '''
        Args:
            drop_path:           The rate of drop_path....
        '''
        super().__init__()
        assert layers % 2 == 0,\
        'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition = PatchMerging(in_dim=dim, out_dim=out_dim)
        else:
            self.patch_partition = None

        num = layers // 2
        self.layers = nn.LayerList()
        for idx in range(num):
            self.layers.append(
                nn.Sequential(
                    Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads,
                          window_size=window_size,shuffle=False,mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          relative_pos_embedding=relative_pos_embedding),
                    Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads,
                          window_size=window_size,shuffle=shuffle,mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop,drop_path=drop_path,
                          relative_pos_embedding=relative_pos_embedding)
                )
            )


    def forward(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x


class PatchEmbedding(nn.Layer):
    '''
    Describe:
        This class shoulde be use to construct some token.
    '''
    def __init__(self,
                 inter_channel=32,
                 out_channel=48):
        '''
            inter_channel:         The input channel of PatchEmbedding class.
            out_channel:           The output channel of PatchEmbedding class.
        '''
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, inter_channel, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2D(inter_channel, 0.1),
            nn.ReLU6()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(inter_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(out_channel, 0.1),
            nn.ReLU6()
        )
        self.conv3 = nn.Conv2D(out_channel, out_channel, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def get_sinusoid_encoding_table(n_position, d_model):
    '''
    Describe:
        This function should be used to construct a NLP style Transformer postition embedding...
    '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return paddle.to_tensor(sinusoid_table, dtype='float32').detach()


class ShuffleTransformer(nn.Layer):
    '''
    Describe:
        The top level class of ShuffleTransformer.
    '''
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 token_dim=32,
                 embed_dim=96,
                 mlp_ratio=4.,
                 layers=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 relative_pos_embedding=True,
                 shuffle=True,
                 window_size=7,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 has_pos_embed=False,
                 **kwargs):
        '''
        Args:
            img_size:             Input image size.
            in_chans:             Input channel number.
            num_classes:          The output class number.
            token_dim:            The dimemsion of token.
            embed_dim:            The dimension of embedding layer.
            layers:               Each block's layer number.
        '''
        super().__init__()
        self.num_classes   = num_classes
        self.num_features  = self.embed_dim = embed_dim
        self.has_pos_embed = has_pos_embed
        dims = [i*32 for i in num_heads]

        self.to_token = PatchEmbedding(inter_channel=token_dim, out_channel=embed_dim)

        num_patches   = (img_size**2) // 16

        if self.has_pos_embed:
            pos_embed_tensor = get_sinusoid_encoding_table(n_position=num_patches,
                                                           d_model=embed_dim)
            pos_attr         = paddle.ParamAttr(initializer=\
                                                paddle.nn.initializer.Assign(pos_embed_tensor))
            self.pos_embed   = paddle.create_parameter(shape=pos_embed_tensor.shape,
                                                       dtype='float32',
                                                       attr=pos_attr)
            self.pos_embed.requires_grad=False
            self.pos_drop    = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, 4)]
        self.stage1 = StageModule(layers[0], embed_dim, dims[0], num_heads[0],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[0],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[1],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[2],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[3],
                                  relative_pos_embedding=relative_pos_embedding)

        self.avgpool= nn.AdaptiveAvgPool2D((1, 1))

        self.head   = nn.Linear(dims[3], num_classes)


    def forward_features(self, x):
        '''
        Describe:
            This function used to forward the feature of Shuffle Transformer.
        Args:
            x --------------> A tensor, whose shape is[n,3,h,w]
        '''
        x = self.to_token(x)
        b, c, h, w = x.shape
        if self.has_pos_embed:
            pos_embed = paddle.reshape(self.pos_embed, shape=[1, h, w, c])
            pos_embed = paddle.transpose(pos_embed, perm=[0, 3, 1, 2])
            x = x + pos_embed
            x = self.pos_drop(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        return x



    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model  = ShuffleTransformer(img_size=224)
    print(model)
