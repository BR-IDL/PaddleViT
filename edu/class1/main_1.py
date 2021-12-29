# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
import paddle
import numpy as np
from PIL import Image

paddle.set_device('cpu')

def main():
    # 1. Create a Tensor
    t = paddle.zeros([3, 3])
    print(t)

    # 2. Create a random Tensor
    t = paddle.randn([4, 3])
    print(t)

    # 3. Create a tensor from Image ./724.jpg 28x28
    img = np.array(Image.open('./724.jpg'))
    for i in range(28):
        for j in range(28):
            print(f'{img[i, j]:03} ', end='')
        print()
    t = paddle.to_tensor(img, dtype='float32')

    # 4. print tensor type and dtype of tensor
    print(type(t))
    print(t.dtype)
    
    # 5. transpose image tensor
    t = t.transpose([1, 0])
    for i in range(28):
        for j in range(28):
            print(f'{int(t[i, j]):03} ', end='')
        print()

    # 6. Reshape a random int tensor from 5x5 to 25
    t = paddle.randint(0, 10, [5, 5])
    print(t)
    t1 = t.reshape([25])
    t2 = t.flatten(0)
    print(t1)
    print(t2)

    # 7. Unsqueeze a random int tensor from 5x5 to 5x5x1
    t = paddle.randint(0, 10, [5, 5])
    print(t)
    print(t.shape)
    print(t.unsqueeze(-1).shape)
    
    # 8. chunk a random int tensor from 5x15 to 5x5, 5x5 and 5x5
    t = paddle.randint(0, 10, [5, 15])
    print(t)
    qkv = t.chunk(3, -1)
    print(type(qkv))
    q, k, v = qkv
    print(q)

if __name__ == "__main__":
    main()
