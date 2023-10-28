from PIL import Image, ImageDraw, ImageChops
import numpy as np
from random import randint
from dataclasses import dataclass
from operator import add
import math
from copy import deepcopy

class Gimg:
    @dataclass
    class Box:
        color: tuple
        shape: list

    def __random_color(self):
        return [randint(0,255), randint(0,255), randint(0,255)]

    def __random_box(self, radius):
        color = self.__random_color()
        center = [randint(0,self.w), randint(0,self.h)]
        size = [randint(0,radius), randint(0,radius)]
        shape = [a - b for a, b in zip(center, size)] \
            + [a + b for a, b in zip(center, size)] 
        return self.Box(color = color, shape = shape)
    
    def __init__(self, width, height, boxes):
        self.w = width
        self.h = height
        self.bg = self.__random_color()
        self.boxes = [ self.__random_box(16) for _ in range(boxes)]

    def permute(self, color_scale, sz_scale):
        other = deepcopy(self)
        other.bg = [
            np.clip(c + randint(-color_scale, color_scale), 0, 255)
            for c in other.bg
        ]

        for box in other.boxes:
            box.color = [
                np.clip(c + randint(-color_scale, color_scale), 0, 255)
                for c in box.color
            ]
            box.shape = [ 
                coord + randint(-sz_scale, sz_scale)
                for coord in box.shape
            ]
            if box.shape[0] > box.shape[2]:
                box.shape[0] = box.shape[2]
            if box.shape[1] > box.shape[3]:
                box.shape[1] = box.shape[3]

        return other

    def as_image(self):
        img = Image.new('RGB', (self.w, self.h), tuple(self.bg))
        draw = ImageDraw.Draw(img)
        for box in self.boxes:
            draw.ellipse(box.shape, fill = tuple(box.color))
        return img

def _loss(a, b):
    diff = np.asarray(ImageChops.difference(a,b)) / 255
    return math.sqrt(np.mean(np.square(diff)))

def generate(img, threshold, color_scale, size_scale):
    gimg = Gimg(img.width, img.height, 64)
    loss = _loss(gimg.as_image(), img)
    i = 0
    while(loss > threshold):
        print("Iteration: ", i)
        print("Loss: ", loss)
        i=i+1
        new_gimg = gimg.permute(color_scale, size_scale)
        new_img = new_gimg.as_image()
        new_loss = _loss(img, new_img)
        if new_loss < loss:
            gimg = new_gimg
            loss = new_loss
    return gimg

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 5:
        print("Usage: gimg filename threshold width height")
    img = Image.open(argv[1])
    img = img.convert("RGB")
    g = generate(img, float(argv[2]), int(argv[3]), int(argv[4]))
    im = g.as_image()
    im.save("image.png", "PNG")
