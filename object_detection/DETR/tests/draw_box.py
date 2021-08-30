import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw

im = np.ones([600, 800, 3])* 255.0
im = im.astype(np.uint8)
im = Image.fromarray(im)

draw = ImageDraw.Draw(im)

boxes = [[100, 100, 300, 400],
         [120, 160, 230, 340],
         [200, 320, 500, 580],
         [400, 450, 700, 550],
         [450, 80, 580, 210]]
color = ['red','blue','magenta','green', 'gold']
xy = [[180, 120],[150, 160],[350, 400],[600, 500],[500, 110]]


for idx, box in enumerate(boxes):
    draw.rectangle(box, fill=None, outline=color[idx], width=4)
    draw.text((box[0],box[1]), f'{box[0]},{box[1]}', fill=color[idx])
    draw.text((box[2],box[3]), f'{box[2]},{box[3]}', fill=color[idx])

    draw.text((xy[idx][0], xy[idx][1]), f'{idx+1}', fill=color[idx])

im.save('box.png')



