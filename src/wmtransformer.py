import fastai
from fastai.vision import *
from fastai.callbacks import *
from PIL import Image, ImageOps
import random as rand
import os
from pathlib import Path
# Inspired from: https://tinyurl.com/y59gyjc7

class oldMold(object):
    def __init__(self, path_lr, path_hr, path_filter):

        """
        :type path_lr: str
        :type path_hr: str
        :type path_filter: str
        """
        
        self.path_lr = path_lr
        self.path_hr = path_hr
        self.path_filter = path_filter

    # https://tinyurl.com/y4jlmqpg
    @staticmethod
    def vec_len(vec):
        return (vec[0] ** 2 + vec[1] ** 2) ** 0.5

    @classmethod
    def pt_dist(cls, pt1, pt2):
        return cls.vec_len((pt1[0] - pt2[0], pt1[1] - pt2[1]))

    @staticmethod
    def clamp(val, minimum, maximum):
        return max(min(val, maximum), minimum)

    @classmethod
    def warp(cls, image, points):
        w_img = Image.new('RGBA', image.size, (255, 0, 0, 0))
        image_pixels = image.load()
        w_img_pixels = w_img.load()
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                offset = [0, 0]
                for point in points:
                    point_position = (point[0] + point[2], point[1] + point[3])
                    shift_vector = (point[2], point[3])
                    helper = 1.5 / (3 * (cls.pt_dist((x, y), point_position) / cls.vec_len(shift_vector)) ** 4 + 1)
                    offset[0] -= helper * shift_vector[0]
                    offset[1] -= helper * shift_vector[1]
                coords = (
                cls.clamp(x + int(offset[0]), 0, image.size[0] - 1), cls.clamp(y + int(offset[1]), 0, image.size[1] - 1))
                w_img_pixels[x, y] = image_pixels[coords[0], coords[1]]
        return w_img

    def __call__(self, fn, i):

        """
        :type filename: str
        :type i: int
        """

        # Create dir if needed and load in images to Resize and Process
        dest = self.path_lr / fn.relative_to(self.path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 250, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w, h = img.size

        # Randomly choose a Damage Augment and Overlay
        total_layers = os.listdir(self.path_filter)
        dmg_num = len(total_layers)
        damage = ('DC' + str(rand.randint(1, dmg_num)) + '.png')
        dmg_name = self.path_filter / damage
        angle = 90 * rand.randint(0, 3)
        top_layer = Image.open(dmg_name).rotate(angle)
        if rand.randint(0, 1) == 1:
            top_layer = ImageOps.mirror(top_layer)

        # Warp damage frame by two vectors v1, v2 - Able to add more warp vectors here as well
        v1 = (rand.randint(-200, -50), rand.randint(50, 200), rand.randint(-200, -50), rand.randint(50, 200))
        v2 = (rand.randint(-200, -50), rand.randint(50, 200), rand.randint(-200, -50), rand.randint(50, 200))
        top_layer = self.warp(top_layer, [v1, v2])

        bottom_layer = img
        bottom_layer.paste(top_layer, (0, 0), top_layer)
        bottom_layer.save(dest, "PNG")
