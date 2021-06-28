import paddle
import paddle.nn as nn
import copy
import math


# DONE
class Mlp(nn.Layer):
    def __init__(self, d_model, dim_feedforward, dropout, act='relu'):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.linear1 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr=w_attr_1,
                                 bias_attr=b_attr_1)

        self.dropout = nn.Dropout(dropout)

        w_attr_2, b_attr_2 = self._init_weights()
        self.linear2 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr=w_attr_2,
                                 bias_attr=b_attr_2)
        if act == 'relu':
            self.act = nn.ReLU()

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# DONE
class Attention(nn.Layer):
    def __init__(self, d_model, n_head, dropout=0.):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.head_dim = int(d_model / n_head)
        self.all_head_dim = self.head_dim * self.n_head
        self.scales = self.head_dim ** -0.5

        w_attr_1, b_attr_1 = self._init_weights()
        self.q = nn.Linear(d_model,
                           self.all_head_dim,
                           weight_attr=w_attr_1,
                           bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.k = nn.Linear(d_model,
                           self.all_head_dim,
                           weight_attr=w_attr_2,
                           bias_attr=b_attr_2)
        w_attr_3, b_attr_3 = self._init_weights()
        self.v = nn.Linear(d_model,
                           self.all_head_dim,
                           weight_attr=w_attr_3,
                           bias_attr=b_attr_3)

        #w_attr, b_attr = self._init_weights()
        #self.qkv = nn.Linear(d_model,
        #                     self.all_head_dim * 3,
        #                     weight_attr=w_attr,
        #                     bias_attr=b_attr)


        w_attr_4, b_attr_4 = self._init_weights()
        self.fc = nn.Linear(self.all_head_dim,
                            d_model,
                            weight_attr=w_attr_4,
                            bias_attr=b_attr_4)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(axis=-1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())
        return weight_attr, bias_attr

    def transpose_for_scores(self, x):
        # [seq_l, batch, all_head_dim] -> [seq_l, batch, n_head, head_dim]
        new_x_shape = x.shape[:-1] + [self.n_head, self.head_dim]
        x = x.reshape(new_x_shape)
        # [seq_l, batch, n_head, head_dim] -> [seq_l, batch*n_head, head_dim]
        x = x.flatten(start_axis=1, stop_axis=2)
        # [seq_l, batch*n_head, head_dim] -> [batch*n_head, seq_l, head_dim]
        x = x.transpose([1, 0, 2])
        return x

    def forward(self, query, key, value, key_pad_mask=None):
        SRC_L = key.shape[0] # key: [seq_l, batch_size, hidden_dim]
        B = key.shape[1]
        TGT_L = query.shape[0]
        EMBED_DIM = query.shape[2]


        attn_mask = None
        if key_pad_mask is not None:
            assert key_pad_mask.shape == [B, SRC_L], f'expecting key_pad_mask shape of {[B, L]}, but got {key_pad_mask.shape}'
            key_pad_mask = key_pad_mask.reshape([B, 1, 1, SRC_L])
            key_pad_mask = key_pad_mask.expand([B, self.n_head, 1, SRC_L])
            key_pad_mask = key_pad_mask.reshape([B*self.n_head, 1, SRC_L])
    
            attn_mask = paddle.zeros_like(key_pad_mask)
            inf_tensor = paddle.ones_like(key_pad_mask) * float('-inf')
            attn_mask = paddle.where(key_pad_mask > 0.5, inf_tensor, attn_mask) # TODO: check True/False

        #print('query shape:', query.shape)
        #x = paddle.concat([query, key, value], axis=-1)
        #print('X shape=', x.shape)
        #qkv = self.qkv(x).chunk(3, axis=-1)
        #q, k, v = map(self.transpose_for_scores, qkv)
        q = self.transpose_for_scores(self.q(query))
        k = self.transpose_for_scores(self.k(key))
        v = self.transpose_for_scores(self.v(value))
        #print('imhere')
        #print('imhere')
        #print('imhere')
        #print('imhere')
        #print('imhere')
        #print('imhere')
        #print('q.w:', self.q.weight )
        #print('q.b:', self.q.bias )
        #print('k.w:', self.k.weight )
        #print('k.b:', self.k.bias )
        #print('v.w:', self.v.weight )
        #print('v.b:', self.v.bias )

        #print('========= q before scaling ========')
        #print(q)
        q = q * self.scales
        #print('========= q after scaling ========')
        #print(q)

        attn = paddle.matmul(q, k, transpose_y=True)


        #print('attn shape=', attn.shape)
        #attn = attn * self.scales
        #print('============ attn =============')
        #print(attn)
        # add mask (-inf) to filter out pad/attn positions
        #print('attn_mask, ', attn_mask.shape)
        #print('attn, ', attn.shape)
        if attn_mask is not None:
            attn += attn_mask
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        #print('======= attn ========')
        #print(attn)

        z = paddle.matmul(attn, v)   # [batch*n_head, seq_l, head_dim]
        #print('======== z =========')
        #print(z)
        z = z.transpose([1, 0, 2]) #[seq_l, batch*n_head, head_dim]
        z = z.reshape([TGT_L, B, EMBED_DIM])

        z = self.fc(z)
        #print('========== z fc =========')
        #print(z)
        z = self.dropout(z)
        return z
        
# DONE
class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, n_head, dropout=dropout)
        self.mlp = Mlp(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.normalize_before = normalize_before # TODO: add pre and post

    def forward_post(self,
                     src,
                     src_key_pad_mask=None,
                     pos=None):
        # + positional embedding to q and k
        #print('src shape=', src.shape)
        #print('pos shape=', pos.shape)
        #print('-------- src and pos')
        #print('src = ', src)
        #print('pos = ', pos)
        #print('------------------------ encoderlayer ----------')
        q = (src + pos) if pos is not None else src
        k = (src + pos) if pos is not None else src
        #print('----- q and k:')
        #print(q)
        #print(k)
        #print(src_key_pad_mask)
        # attention
        src2 = self.self_attn(query=q, key=k, value=src, key_pad_mask=src_key_pad_mask)

        #print('----- src2:')

        # attention add & norm
        src = src + src2
        #print('==== src before norm1')
        #print(src)
        src = self.norm1(src)
        #print('==== src after norm1')
        #print(src)
        # FFN
        src2 = self.mlp(src)
        #print('===== src2 ')
        #print(src2)
        # FFN add & norm
        src = src + src2
        src = self.norm2(src)
        return src

    def forward(self, src, src_key_pad_mask=None, pos=None):
        return self.forward_post(src, src_key_pad_mask, pos)
        

# DONE
class TransformerEncoder(nn.Layer):
    def __init__(self, layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_pad_mask=None, pos=None):
        output = src
        for idx, layer in enumerate(self.layers):
            #print(f'---------- encoder {idx} ------------')
            output = layer(output, src_key_pad_mask=src_key_pad_mask, pos=pos)
            #print(output, output.shape)
        if self.norm is not None:
            output = self.norm(output)
        #print(f'---------- last encoder after norm ------------')
        #print(output, output.shape)
        return output





class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 n_head,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = Attention(d_model, n_head, dropout=dropout)
        self.dec_enc_attn = Attention(d_model, n_head, dropout=dropout)
        self.mlp = Mlp(d_model, dim_feedforward, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.normalize_before = normalize_before #TODO: add forward_pre and post

    def forward_post(self,
                     tgt,
                     memory,
                     memory_key_pad_mask=None,
                     pos=None,
                     query_pos=None):
        # + positional embedding to q and k
        q = (tgt + query_pos) if query_pos is not None else tgt
        k = (tgt + query_pos) if query_pos is not None else tgt
        # dec self attention
        #print('----- decoder self_attn: ')
        tgt2 = self.self_attn(query=q,
                              key=k,
                              value=tgt) 
        #print('================================')
        #print('===========dec self_attn =================')
        #print('================================')
        #print(tgt2)

        # dec self attention add & norm
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        # dec enc attention 
        tgt2 = self.dec_enc_attn(query=(tgt + query_pos) if query_pos is not None else tgt,
                                 key=(memory + pos) if pos is not None else memory,
                                 value=memory,
                                 key_pad_mask=memory_key_pad_mask)

        #print('================================')
        #print('===========dec dec_enc_attn==================')
        #print('================================')
        #print(tgt2)

        # dec enc attention add & norm
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.mlp(tgt)
        # FFN add & norm
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, memory_key_pad_mask=None, pos=None, query_pos=None):
        return self.forward_post(tgt, memory, memory_key_pad_mask, pos, query_pos)


class TransformerDecoder(nn.Layer):
    def __init__(self, layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.LayerList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                target,
                memory,
                memory_key_pad_mask=None,
                pos=None,
                query_pos=None):
        output = target
        intermediate = []

        for idx, layer in enumerate(self.layers):
            #print(f'---------- decoder {idx} ------------')
            output = layer(output,
                           memory,
                           memory_key_pad_mask=memory_key_pad_mask,
                           pos=pos,
                           query_pos=query_pos)
            #print(output, output.shape)
            if self.return_intermediate:
                #print(output, output.shape)
                #print(self.norm.weight)
                #print(self.norm.bias)
                #print('-------------- before and after norm --------------')
                #print(self.norm(output), self.norm(output).shape)
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            #print('!!!!!!!!!!!!!!!!!!!')
            #print(intermediate)
            return paddle.stack(intermediate)

        #print('!!!!!!!!!!!!!!!!!!!')
        #print(output, output.shape)
        return output.unsqueeze(0)


class Transformer(nn.Layer):
    def __init__(self,
                d_model=256,
                n_head=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
                normalize_before=False,
                return_intermediate_dec=True):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model,
                                                n_head,
                                                dim_feedforward,
                                                dropout,
                                                activation,
                                                normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer,
                                          num_encoder_layers,
                                          encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model,
                                                n_head,
                                                dim_feedforward,
                                                dropout,
                                                activation,
                                                normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers,
                                          decoder_norm,
                                          return_intermediate_dec)
        
        self._reset_params()

        self.n_head = n_head
        self.d_model = d_model

    def _reset_params(self):
        pass

    def forward(self, src, mask, query_embed, pos_embed):
        B, C, H, W = src.shape
        src = src.flatten(2) # [B, C, H, W] -> [B, C, H*W]
        src = src.transpose([2, 0, 1]) # [B, C, H*W] -> [H*W, B, C]
        pos_embed = pos_embed.flatten(2) # [B, dim, H, W] -> [B, dim, H*W]
        pos_embed = pos_embed.transpose([2, 0, 1]) # [B, dim, H*W] -> [H*W, B, dim]
        query_embed = query_embed.unsqueeze(1) #[num_queries, 1, d_model]
        query_embed = query_embed.expand((query_embed.shape[0], B, query_embed.shape[2]))
        mask = mask.flatten(1) # this mask is batch mask for multiple image sizes

        target = paddle.zeros_like(query_embed) # decoder 1st input is set to all zeros


        #print('----- inside transformer')
        #print(src.shape)
        #print(pos_embed.shape)
        #print(query_embed.shape)
        #print(mask.shape)
        #print('-----')

        memory = self.encoder(src, src_key_pad_mask=mask, pos=pos_embed)

        #print('||||||||||||||| memory |||||||||||||')
        #print(memory, memory.shape)

        hs = self.decoder(target,
                          memory,
                          memory_key_pad_mask=mask,
                          pos=pos_embed,
                          query_pos=query_embed)
        
        #print('hs shape:', hs.shape)
        #print(hs)
        hs = hs.transpose([0, 2, 1, 3])  # [1, batch, n_queries, embed_dim]
        memory = memory.transpose([1, 2, 0])
        memory = memory.reshape([B, C, H, W])

        return hs, memory
        






