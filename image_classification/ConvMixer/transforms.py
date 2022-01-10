import random
import paddle
import paddle.nn
import paddle.vision.transforms as T


class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return T.hflip(image) 
        return image
