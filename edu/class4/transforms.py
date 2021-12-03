import numpy as np
from PIL import Image
import paddle
import paddle.vision.transforms as T
paddle.set_device('cpu')


def crop(image, region):
    # region: [i, j, h, w]
    cropped_image = T.crop(image, *region)
    return cropped_image


class CenterCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        ch, cw = self.size
        crop_top = int(round(h - ch) / 2.)
        crop_left = int(round(w - cw) / 2.)
        return crop(image, (crop_top, crop_left, ch, cw))


class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return T.resize(image, self.size)


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, image):
        w, h = image.size
        img = paddle.to_tensor(np.array(image))
        if img.dtype == paddle.uint8:
            img = paddle.cast(img, 'float32') /255.
        img = img.transpose([2, 0, 1]) # 'CHW'
        return img


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image
    

def main():
    img = Image.open('img.jpg')
    transforms = Compose([Resize([256, 256]), CenterCrop([112, 112])])
    #transforms = Compose([Resize([256, 256]), CenterCrop([112, 112]), ToTensor()])
    out = transforms(img)
    print(out)
    out.save('img_crop.jpg')

if __name__ == "__main__":
    main()
