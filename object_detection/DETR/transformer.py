import copy
import math
import paddle
import paddle.nn as nn

class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        w_attr_1, b_attr_1 = self.init_weights()
        self.fc1 = nn.Linear(embed_dim,
                             int(embed_dim * mlp_ratio),
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = self.init_weights()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio),
                             embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = int(dim / num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scales =  self.head_dim ** -0.5

        w_attr_1, b_attr_1 = self.init_weights()
        self.q = nn.Linear(self.dim,
                           self.all_head_dim,
                           weight_attr=w_attr_1,
                           bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self.init_weights()
        self.k = nn.Linear(self.dim,
                           self.all_head_dim,
                           weight_attr=w_attr_2,
                           bias_attr=b_attr_2)
        w_attr_3, b_attr_3 = self.init_weights()
        self.v = nn.Linear(self.dim,
                           self.all_head_dim,
                           weight_attr=w_attr_3,
                           bias_attr=b_attr_3)

        w_attr_4, b_attr_4 = self.init_weights()
        self.proj = nn.Linear(self.all_head_dim,
                              self.dim,
                              weight_attr=w_attr_4,
                              bias_attr=b_attr_4)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return weight_attr, bias_attr

    def transpose_multihead(self, x):
        # [seq_l, batch, all_head_dim] -> [seq_l, batch, num_heads, head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # [seq_l, batch, num_heads, head_dim] -> [seq_l, batch*num_heads, head_dim]
        x = x.flatten(start_axis=1, stop_axis=2)
        # [seq_l, batch*num_heads, head_dim] -> [batch*num_heads, seq_l, head_dim]
        x = x.transpose([1, 0, 2])
        return x

    def forward(self, query, key, value, key_pad_mask=None):
        key_len = key.shape[0] # when enc-dec attn: num_patches (sequence len, token len)
        batch_size = key.shape[1] # when enc-dec attn: batch_size
        query_len = query.shape[0] # when enc-dec attn: num_queries
        embed_dim = query.shape[2] # when end-dec attn: embed_dim

        attn_mask = None
        if key_pad_mask is not None:
            assert key_pad_mask.shape == [batch_size, key_len]
            key_pad_mask = key_pad_mask.reshape([batch_size, 1, 1, key_len])
            key_pad_mask = key_pad_mask.expand([batch_size, self.num_heads, 1, key_len])
            key_pad_mask = key_pad_mask.reshape([batch_size * self.num_heads, 1, key_len])

            attn_mask = paddle.zeros_like(key_pad_mask)
            inf_tensor = paddle.ones_like(key_pad_mask) * float('-inf')
            attn_mask = paddle.where(key_pad_mask > 0.5, inf_tensor, attn_mask)

        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        q, k, v = map(self.transpose_multihead, [q, k, v])

        q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        out = paddle.matmul(attn, v) # [batch*num_heads, seq_l, head_dim]
        out = out.transpose([1, 0, 2]) #[seq_l, batch*num_heads, head_dim]
        out = out.reshape([query_len, batch_size, embed_dim])

        out = self.proj(out)
        out = self.dropout(out)

        return out


class EncoderLayer(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 dropout=0.0,
                 attention_dropout=0.0,
                 pre_norm=False):
        super().__init__()
        # self attention and self attention layer norm
        w_attr_1, b_attr_1 = self.init_weights()
        self.attn_norm = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(dim, num_heads, attention_dropout)
        # mlp and mlp norm
        w_attr_2, b_attr_2 = self.init_weights()
        self.mlp_norm = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

        self.pre_norm = pre_norm

    def init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return weight_attr, bias_attr

    def forward_pre(self, x, key_pad_mask=None, pos=None):
        # self attention + residual
        h = x
        x = self.attn_norm(x)
        q = x + pos if pos is not None else x
        k = x + pos if pos is not None else x
        x = self.attn(query=q, key=k, value=x, key_pad_mask=key_pad_mask)
        x = x + h

        # mlp + residual
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x

    def forward_post(self, x, key_pad_mask=None, pos=None):
        # self attention + residual
        h = x
        q = x + pos if pos is not None else x
        k = x + pos if pos is not None else x
        x = self.attn(query=q, key=k, value=x, key_pad_mask=key_pad_mask)
        x = x + h
        x = self.attn_norm(x)

        # mlp + residual
        h = x
        x = self.mlp(x)
        x = x + h
        x = self.mlp_norm(x)
        return x

    def forward(self, x, key_pad_mask=None, pos=None):
        if self.pre_norm:
            return self.forward_pre(x, key_pad_mask, pos)
        else:
            return self.forward_post(x, key_pad_mask, pos)


class DecoderLayer(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 dropout=0.0,
                 attention_dropout=0.0,
                 pre_norm=False):
        super().__init__()
        # self attention layer norm
        w_attr_1, b_attr_1 = self.init_weights()
        self.attn_norm = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        # self attention
        self.attn = Attention(dim, num_heads, attention_dropout)
        
        # enc-dec attn layer norm
        w_attr_2, b_attr_2 = self.init_weights()
        self.enc_dec_attn_norm = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        # enc-dec attention
        self.enc_dec_attn = Attention(dim, num_heads, attention_dropout)

        # mlp
        w_attr_3, b_attr_3 = self.init_weights()
        self.mlp_norm = nn.LayerNorm(dim, weight_attr=w_attr_3, bias_attr=b_attr_3)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

        self.pre_norm = pre_norm

    def init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return weight_attr, bias_attr

    def forward_pre(self, x, enc_out, key_pad_mask=None, pos=None, query_pos=None):
        # self attention + residual
        h = x
        x = self.attn_norm(x)
        q = x + query_pos if query_pos is not None else x
        k = x + query_pos if query_pos is not None else x
        x = self.attn(query=q, key=k, value=x)
        x = x + h
        # enc-dec attention + residual
        h = x
        x = self.enc_dec_attn_norm(x)
        q = x + query_pos if query_pos is not None else x
        k = enc_out + pos if pos is not None else enc_out
        x = self.enc_dec_attn(query=q, key=k, value=enc_out, key_pad_mask=key_pad_mask)
        x = x + h
        # mlp + residual
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x

    def forward_post(self, x, enc_out, key_pad_mask=None, pos=None, query_pos=None):
        # self attention + residual
        h = x
        q = x + query_pos if query_pos is not None else x
        k = x + query_pos if query_pos is not None else x
        x = self.attn(query=q, key=k, value=x)
        x = x + h
        x = self.attn_norm(x)
        # enc-dec attention + residual
        h = x
        q = x + query_pos if query_pos is not None else x
        k = enc_out + pos if pos is not None else enc_out
        x = self.enc_dec_attn(query=q, key=k, value=enc_out, key_pad_mask=key_pad_mask)
        x = x + h
        x = self.enc_dec_attn_norm(x)
        # mlp + residual
        h = x
        x = self.mlp(x)
        x = x + h
        x = self.mlp_norm(x)
        return x

    def forward(self, x, enc_out, key_pad_mask=None, pos=None, query_pos=None):
        if self.pre_norm:
            return self.forward_pre(x, enc_out, key_pad_mask, pos, query_pos)
        else:
            return self.forward_post(x, enc_out, key_pad_mask, pos, query_pos)


class Transformer(nn.Layer):
    def __init__(self,
                 dim=512,
                 num_heads=8,
                 num_encoders=6,
                 num_decoders=6,
                 mlp_ratio=4.0,
                 dropout=0.0,
                 attention_dropout=0.0,
                 pre_norm=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.dim = dim
        # create encoder
        encoder_layer_list = []
        for i in range(num_encoders):
            encoder_layer_list.append(EncoderLayer(dim,
                                                   num_heads,
                                                   mlp_ratio,
                                                   dropout,
                                                   attention_dropout,
                                                   pre_norm))
        self.encoder = nn.LayerList(encoder_layer_list)
        # create decoder
        decoder_layer_list = []
        for i in range(num_decoders):
            decoder_layer_list.append(DecoderLayer(dim,
                                                   num_heads,
                                                   mlp_ratio,
                                                   dropout,
                                                   attention_dropout,
                                                   pre_norm))
        self.decoder = nn.LayerList(decoder_layer_list)

        if pre_norm:
            w_attr_1, b_attr_1 = self.init_weights()
            self.encoder_norm = nn.LayerNorm(dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        else:
            self.encoder_norm = None

        w_attr_2, b_attr_2 = self.init_weights()
        self.decoder_norm = nn.LayerNorm(dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        
        self.return_intermediate_dec = return_intermediate_dec

    def init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.))
        return weight_attr, bias_attr

    def forward(self, x, mask, query_embed, pos_embed):
        B, C, H, W = x.shape
        x = x.flatten(2) # [B, C, H, W] -> [B, C, H*W]
        x = x.transpose([2, 0, 1]) # [B, C, H*W] -> [H*W, B, C]

        pos_embed = pos_embed.flatten(2) # [B, dim, H, W] -> [B, dim, H*W]
        pos_embed = pos_embed.transpose([2, 0, 1]) # [B, dim, H*W] -> [H*W, B, dim]

        query_embed = query_embed.unsqueeze(1) #[num_queries, 1, d_model]
        query_embed = query_embed.expand([query_embed.shape[0], B, query_embed.shape[2]])
        mask = mask.flatten(1) # this mask is batch mask for multiple image sizes

        target = paddle.zeros_like(query_embed) # decoder 1st input is set to all zeros
        
        encoder_out = x
        for idx, encoder_layer in enumerate(self.encoder):
            encoder_out = encoder_layer(encoder_out, mask, pos_embed)

        if self.encoder_norm is not None:
            encoder_out = self.encoder_norm(encoder_out)

        intermediate = []
        decoder_out = target
        for idx, decoder_layer in enumerate(self.decoder):
            decoder_out = decoder_layer(decoder_out, encoder_out, mask, pos_embed, query_embed)
            if self.return_intermediate_dec:
                intermediate.append(self.decoder_norm(decoder_out))

        if self.decoder_norm is not None:
            decoder_out = self.decoder_norm(decoder_out)
            if self.return_intermediate_dec:
                intermediate.pop()
                intermediate.append(decoder_out)
        if self.return_intermediate_dec:
            decoder_out = paddle.stack(intermediate)
        else:
            decoder_out = decoder_out.unsqueeze(0)

        encoder_out = encoder_out.transpose([1, 2, 0])
        encoder_out = encoder_out.reshape([B, C, H, W])
        decoder_out = decoder_out.transpose([0, 2, 1, 3]) # [n_layers, batch, n_queries, dim]

        return decoder_out, encoder_out


def build_transformer(config):
    model = Transformer(dim=config.MODEL.TRANS.EMBED_DIM,
                        num_heads=config.MODEL.TRANS.NUM_HEADS,
                        num_encoders=config.MODEL.TRANS.NUM_ENCODERS,
                        num_decoders=config.MODEL.TRANS.NUM_DECODERS,
                        mlp_ratio=config.MODEL.TRANS.MLP_RATIO,
                        dropout=config.MODEL.DROPOUT,
                        attention_dropout=config.MODEL.ATTENTION_DROPOUT,
                        pre_norm=config.MODEL.TRANS.PRE_NORM,
                        return_intermediate_dec=config.MODEL.TRANS.RETURN_INTERMEDIATE_DEC)
    return model


