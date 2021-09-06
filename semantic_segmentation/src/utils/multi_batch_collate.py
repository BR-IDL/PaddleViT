import numpy as np

class multi_val_fn():
    def __init__(self) -> None:
        pass

    def __call__(self, datas) -> tuple:
        img_list = []
        label_list = []
        
        for img, label in datas:
            img_list.append(img)
            label_list.append(label.astype('int64'))

        return img_list, label_list