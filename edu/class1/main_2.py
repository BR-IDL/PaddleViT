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


def main():
    # 1. Load image and convert to tensor
    img = Image.open('./724.jpg')
    img = np.array(img)
    for i in range(28):
        for j in range(28):
            print(f'{img[i,j]:03} ', end='')
        print()

    sample = paddle.to_tensor(img, dtype='float32')
    # simulate a batch of data
    sample = sample.reshape([1, 1, 28, 28])
    print(sample.shape)

    # 2. Patch Embedding
    patch_embedding = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, embed_dim=1)
    out = patch_embedding(sample)
    print(out)
    print(out.shape)
    for i in range(0, 28, 7):
        for j in range(0, 28, 7):
            print(paddle.sum(sample[0, 0, i:i+7, j:j+7]).numpy().item())
    


    patch_embedding = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, embed_dim=96)
    out = patch_embedding(sample)
    # 3. mlp
    mlp = Mlp(96, 4.0)
    out = mlp(out)
    print(out)
    print(out.shape)



if __name__ == "__main__":
    main()
