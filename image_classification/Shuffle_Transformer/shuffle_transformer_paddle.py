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
        A Feed Forward Network layer, which use Conv replace the Linear project.
    '''
    def __init__(self, in_feature, hidden_features=None, out_features=None, drop=0.):
        '''
        Args:
            in_feature:       The MLP's input feature dim.
            hidden_features:  The MLP's hidden feature dim.
            out_features:     The MLP's output feature dim.
            act_layer:        The act function of Conv2D.
            drop:             The drop rate of dropout.
        '''
        super().__init__()
        out_feature = out_features or in_feature
        hid_feature = hidden_features or in_feature
        self.fc1    = nn.Conv2D(in_feature, hid_feature, 1, 1, 0)
        self.act    = nn.ReLU6()
        self.fc2    = nn.Conv2D(hid_feature, out_feature, 1, 1, 0)
        self.drop   = nn.Dropout(drop)

    def forward(self, inputs):
        # input tensor should be [batch_size, hidden_dim, height, width]
        temp = self.fc1(inputs)
        temp = self.act(temp)
        temp = self.drop(temp)
        temp = self.fc2(temp)
        res  = self.drop(temp)
        return res



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

    def shuffle_and_divide_qkv(self, tensor):
        '''
        Describe:
            This function used to replace the einops.
            rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d')
        Args:
            tensor: A paddle tensor with shape [batch_size, hidden_dim * 3, height, width]
        Returns:
            query:  A paddle tensor with shape [batch_size*hidden_dim,head_num,window_area,head_dim]
            key:    A paddle tensor with shape [batch_size*hidden_dim,head_num,window_area,head_dim]
            value:  A paddle tensor with shape [batch_size*hidden_dim,head_num,window_area,head_dim]
        '''
        #b0, c0, h0, w0 = tensor.shape
        origin_b, origin_c, origin_h, origin_w = tensor.shape
        if self.shuffle:
            # reshape the tensor from
            # b,(qkv h d),(ws1 hh),(ws2 ww) --> b, qkv, h, d, ws1, hh, ws2, ww
            tensor         = paddle.reshape(tensor,
                                            shape=[origin_b,
                                                   3, self.num_heads,
                                                   origin_c // (3 * self.num_heads),
                                                   self.ws, origin_h // self.ws,
                                                   self.ws, origin_w // self.ws])
            div_b, _, div_h, div_d, div_ws1, div_hh, div_ws2, div_ww = tensor.shape
            # transpose the tensor to qkv,b,hh,ww,h,ws1,ws2,d
            tensor         = paddle.transpose(tensor, perm=[1, 0, 5, 7, 2, 4, 6, 3])
            # reshape the tensor to qkv,(b hh ww),h,(ws1 ws2),d
            tensor         = paddle.reshape(tensor, shape=[3,
                                                           (div_b * div_hh * div_ww),
                                                           div_h, (div_ws1 * div_ws2),
                                                           div_d])
        else:
            # reshape the tensor from b,
            # (qkv h d),(ws1 hh),(ws2 ww) --> b, qkv, h, d, hh, ws1, ww, ws2
            tensor         = paddle.reshape(tensor,
                                            shape=[origin_b,
                                                   3, self.num_heads,
                                                   origin_c // (3 * self.num_heads),
                                                   origin_h // self.ws, self.ws,
                                                   origin_w // self.ws, self.ws])
            div_b, _, div_h, div_d, div_hh, div_ws1, div_ww, div_ws2 = tensor.shape
            # transpose the tensor to qkv,b,hh,ww,h,ws1,ws2,d
            tensor         = paddle.transpose(tensor, perm=[1, 0, 4, 6, 2, 5, 7, 3])
            # reshape the tensor to qkv,(b hh ww),h,(ws1 ws2),d
            tensor         = paddle.reshape(tensor, shape=[3,
                                                           (div_b * div_hh * div_ww),
                                                           div_h, (div_ws1 * div_ws2),
                                                           div_d])
        query, key, value  = paddle.unbind(tensor, axis=0)
        return query, key, value

    def flatten_2_image(self, tensor, origin_shape):
        '''
        Describe:
            This function used to replace the einops's rearrange function.
            Rearrange(Input, '(batch_size height width) head_number (window_size1 window_size2)
            head_dim -> batch_size (head_number head_dim) (window_s1 height) (window_s2 width)')
        Args:
            tensor:         A paddle tensor with shape [batch_size * window_area,
                            head_num, window_size, head_dim].
            origin_shape:   The original input tensor's shape. It should be [batch_size,
                            hidden_dim, window_h, window_w].
        Return:
            result:         A paddle tensor with shape [batch_size, hidden_dim, window_h, window_w]
        '''
        origin_b, origin_c, origin_h, origin_w = origin_shape
        if self.shuffle:
            # reshape the tensor to b, hh, ww, h, ws1, ws2, d
            tensor     = paddle.reshape(tensor,
                                        shape=[origin_b,
                                               origin_h // self.ws,
                                               origin_w // self.ws,
                                               self.num_heads,
                                               self.ws, self.ws,
                                               origin_c // self.num_heads])
            div_b, div_hh, div_ww, div_h, div_ws1, div_ws2, div_d = tensor.shape
            # reshape the tensor to b, h, d, ws1, hh, ws2, ww
            tensor     = paddle.transpose(tensor, perm=[0, 3, 6, 4, 1, 5, 2])
            result     = paddle.reshape(tensor,
                                        shape=[div_b,
                                               (div_h * div_d),
                                               (div_ws1 * div_hh),
                                               (div_ws2 * div_ww)])
        else:
            # reshape the tensor to b, hh, ww, h, ws1, ws2, d
            tensor     = paddle.reshape(tensor,
                                        shape=[origin_b,
                                               origin_h // self.ws,
                                               origin_w // self.ws,
                                               self.num_heads, self.ws, self.ws,
                                               origin_c // self.num_heads])
            div_b, div_hh, div_ww, div_h, div_ws1, div_ws2, div_d = tensor.shape
            # reshape the tensor to b, h, d, hh, ws1, ww, ws2
            tensor     = paddle.transpose(tensor, perm=[0, 3, 6, 1, 4, 2, 5])
            result     = paddle.reshape(tensor,
                                        shape=[div_b,
                                               (div_h * div_d),
                                               (div_hh * div_ws1),
                                               (div_ww * div_ws2)])
        return result


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
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias


    def forward(self, inputs):
        origin_shape = inputs.shape
        qkv          = self.to_qkv(inputs)
        query, key, value      = self.shuffle_and_divide_qkv(qkv)

        dot          = paddle.matmul(query, paddle.transpose(key, perm=[0, 1, 3, 2])) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.get_relative_pos_bias_from_pos_index()
            relative_position_bias = paddle.reshape(relative_position_bias,
                                                    shape=[self.ws * self.ws,
                                                           self.ws * self.ws, -1])
            relative_position_bias = paddle.transpose(relative_position_bias, perm=[2, 0, 1])
            dot      += paddle.unsqueeze(relative_position_bias, axis=0)

        attn         = self.softmax(dot)
        out          = paddle.matmul(attn, value)

        out          = self.flatten_2_image(out, origin_shape)

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
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.BatchNorm2D,
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
                                drop=drop)
        self.norm3        = norm_layer(out_dim, 0.1)



    def forward(self, inputs):
        temp = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        temp = temp + self.local(self.norm2(temp))
        res  = temp + self.drop_path(self.mlp(self.norm3(temp)))
        return res

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

    def forward(self, inputs):
        temp = self.norm(inputs)
        res  = self.reduction(temp)
        return res

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
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
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
                          qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                          relative_pos_embedding=relative_pos_embedding),
                    Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads,
                          window_size=window_size,shuffle=shuffle,mlp_ratio=mlp_ratio,
                          qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop,drop_path=drop_path,
                          relative_pos_embedding=relative_pos_embedding)
                )
            )


    def forward(self, inputs):
        if self.patch_partition:
            inputs = self.patch_partition(inputs)
        temp = inputs
        for regular_block, shifted_block in self.layers:
            temp = regular_block(temp)
            temp = shifted_block(temp)
        return temp


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


    def forward(self, inputs):
        temp = self.conv1(inputs)
        temp = self.conv2(temp)
        res  = self.conv3(temp)
        return res


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
                 num_classes=1000,
                 token_dim=32,
                 embed_dim=96,
                 mlp_ratio=4.,
                 layers=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 relative_pos_embedding=True,
                 shuffle=True,
                 window_size=7,
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
                                  mlp_ratio=mlp_ratio,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[0],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[1],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[2],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3],
                                  window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio,
                                  qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[3],
                                  relative_pos_embedding=relative_pos_embedding)

        self.avgpool= nn.AdaptiveAvgPool2D((1, 1))

        self.head   = nn.Linear(dims[3], num_classes)


    def forward_features(self, inputs):
        '''
        Describe:
            This function used to forward the feature of Shuffle Transformer.
        Args:
            inputs --------------> A tensor, whose shape is[n,3,h,w]
        '''
        temp = self.to_token(inputs)
        _, channel, height, width = temp.shape
        if self.has_pos_embed:
            pos_embed = paddle.reshape(self.pos_embed, shape=[1, height, width, channel])
            pos_embed = paddle.transpose(pos_embed, perm=[0, 3, 1, 2])
            temp = temp + pos_embed
            temp = self.pos_drop(temp)
        temp = self.stage1(temp)
        temp = self.stage2(temp)
        temp = self.stage3(temp)
        temp = self.stage4(temp)
        temp = self.avgpool(temp)
        result = paddle.flatten(temp, 1)
        return result



    def forward(self, inputs):
        temp = self.forward_features(inputs)
        res  = self.head(temp)
        return res


if __name__ == '__main__':
    model  = ShuffleTransformer(img_size=224)
    print(model)
