# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
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


class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                         bias_attr=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [n, c, h, w]
        x = self.patch_embedding(x) # [n, c', h', w']
        x = x.flatten(2) # [n, c', h'*w']
        x = x.transpose([0, 2, 1]) # [n, h'*w', c']
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention()
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)

    def forward(self, x):
        h = x 
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)
        layer_list = [EncoderLayer(16) for i in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x) # [n, h*w, c]: 4, 1024, 16
        for encoder in self.encoders:
            x = encoder(x)
        # avg
        x = x.transpose([0, 2, 1])
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


def main():
    t = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(t)
    print(out.shape)


if __name__ == "__main__":
    main()
