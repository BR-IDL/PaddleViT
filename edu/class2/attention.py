import paddle
import paddle.nn as nn

paddle.set_device('cpu')

class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads,
                 qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.softmax = nn.Softmax(-1)

    def transpose_multi_head(self, x):
        # x:[n, num_patches, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # x:[n, num_patches, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x:[n, num_heads, num_patches, head_dim]
        return x

    def forward(self, x):
        B, N, _ = x.shape
        # x: [n, num_patches, embed_dim]
        qkv = self.qkv(x).chunk(3, -1)
        # qkv:  [n, num_patches, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)
        # q, k, v:[n, num_heads, num_patches, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.scale * attn
        attn = self.softmax(attn)
        attn_weights = attn
        attn = self.attention_dropout(attn)
        # attn: [n, num_heads, num_patches, num_patches]
    
        out = paddle.matmul(attn, v)
        # out: [n, num_heads, num_patches, head_dim]
        out = out.transpose([0, 2, 1, 3])
        # out: [n, num_patches, num_heads, head_dim]
        out = out.reshape([B, N, -1])

        out = self.proj(out)
        out = self.dropout(out)
        return out, attn_weights

def main():
    t = paddle.randn([4, 16, 96])
    print('input shape = ', t.shape)

    model = Attention(embed_dim=96, num_heads=8, 
                      qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.)
    print(model)

    out, attn_weights = model(t)
    print(out.shape)
    print(attn_weights.shape)


if __name__ == "__main__":
    main()
