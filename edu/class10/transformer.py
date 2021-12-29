# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https:///github.com/BR-IDL/PaddleViT)
# 2021.11

import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    """multi-head self attention"""
    def __init__(self, embed_dim, num_heads, qkv_bias=True, dropout=0., attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.scales = self.head_dim ** -0.5


        # CLASS 10: support decoder
        self.q = nn.Linear(embed_dim,
                           self.all_head_dim)
        self.k = nn.Linear(embed_dim,
                           self.all_head_dim)
        self.v = nn.Linear(embed_dim,
                           self.all_head_dim)


        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        # x: [seq_l, batch, all_head_dim] -> [seq_l, batch, n_head, head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        x = x.flatten(1, 2) # merge batch and n_head:  [seq_l, batch*n_head, head_dim]
        x = x.transpose([1, 0, 2]) #[batch * n_head, seq_l, head_dim]
        return x

    def forward(self, query, key, value):
        lk = key.shape[0] # when enc-dec: num_patches (sequence len, token len)
        b = key.shape[1] # when enc-dec: batch_size
        lq = query.shape[0] # when enc-dec: num_queries
        d = query.shape[2] # when enc-dec: embed_dim
    
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        q, k, v = map(self.transpose_multihead, [q, k, v])

        print(f'----- ----- ----- ----- [Attn] batch={key.shape[1]}, n_head={self.num_heads}, head_dim={self.head_dim}')
        print(f'----- ----- ----- ----- [Attn] q: {q.shape}, k: {k.shape}, v:{v.shape}')
        attn = paddle.matmul(q, k, transpose_y=True) # q * k'
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)
        print(f'----- ----- ----- ----- [Attn] attn: {attn.shape}')

        out = paddle.matmul(attn, v)
        out = out.transpose([1, 0, 2])
        out = out.reshape([lq, b, d])

        out = self.proj(out)
        out = self.dropout(out)

        return out


class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)

    def forward(self, x, pos=None):

        h = x  
        x = self.attn_norm(x)
        q = x + pos if pos is not None else x
        k = x + pos if pos is not None else x
        print(f'----- ----- ----- encoder q: {q.shape}, k: {k.shape}, v:{x.shape}')
        x = self.attn(q, k, x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        print(f'----- ----- ----- encoder out: {x.shape}')
        return x


class DecoderLayer(nn.Layer):
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.enc_dec_attn_norm = nn.LayerNorm(embed_dim)
        self.enc_dec_attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)

    def forward(self, x, enc_out, pos=None, query_pos=None):

        h = x  
        x = self.attn_norm(x)
        q = x + query_pos if pos is not None else x
        k = x + query_pos if pos is not None else x
        print(f'----- ----- ----- decoder(self-attn) q: {q.shape}, k: {k.shape}, v:{x.shape}')
        x = self.attn(q, k, x)
        x = x + h

        h = x  
        x = self.enc_dec_attn_norm(x)
        q = x + query_pos if pos is not None else x
        k = enc_out + pos if pos is not None else x
        v = enc_out
        print(f'----- ----- ----- decoder(enc-dec attn) q: {q.shape}, k: {k.shape}, v:{v.shape}')
        x = self.attn(q, k, v)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        print(f'----- ----- ----- decoder out: {x.shape}')
        return x


class Transformer(nn.Layer):
    def __init__(self, embed_dim=32, num_heads=4, num_encoders=2, num_decoders=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.LayerList([EncoderLayer(embed_dim, num_heads) for i in range(num_encoders)])
        self.decoder = nn.LayerList([DecoderLayer(embed_dim, num_heads) for i in range(num_decoders)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, query_embed, pos_embed):
        B, C, H, W = x.shape
        print(f'----- ----- Transformer INPUT: {x.shape}')
        x = x.flatten(2) #[B, C, H*W]
        x = x.transpose([2, 0, 1]) # [H*W, B, C]
        print(f'----- ----- Transformer INPUT(after reshape): {x.shape}')

        # [B, dim, H, W]
        pos_embed = pos_embed.flatten(2)
        pos_embed = pos_embed.transpose([2, 0, 1]) #[H*W, B, dim]
        print(f'----- ----- pos_embed(after reshape): {pos_embed.shape}')

        # [num_queries, dim]
        query_embed = query_embed.unsqueeze(1)
        query_embed = query_embed.expand((query_embed.shape[0], B, query_embed.shape[2]))
        print(f'----- ----- query_embed(after reshape): {query_embed.shape}')

        target = paddle.zeros_like(query_embed)
        print(f'----- ----- target (now all zeros): {target.shape}')

        for encoder_layer in self.encoder:
            encoder_out = encoder_layer(x, pos_embed)
        encoder_out = self.encoder_norm(encoder_out)
        print(f'----- ----- encoder out: {encoder_out.shape}')

        for decoder_layer in self.decoder:
            decoder_out = decoder_layer(target,
                                        encoder_out,
                                        pos_embed,
                                        query_embed)
        decoder_out = self.decoder_norm(decoder_out)
        decoder_out = decoder_out.unsqueeze(0)
        print(f'----- ----- decoder out: {decoder_out.shape}')


        decoder_out = decoder_out.transpose([0, 2, 1, 3]) #[1, B, num_queries, embed_dim]
        encoder_out = encoder_out.transpose([1, 2, 0])
        encoder_out = encoder_out.reshape([B, C, H, W])
        print(f'----- ----- decoder out(after reshape): {decoder_out.shape}')

        return decoder_out, encoder_out


def main():
    trans = Transformer()
    print(trans)


if __name__ == "__main__":
    main()

