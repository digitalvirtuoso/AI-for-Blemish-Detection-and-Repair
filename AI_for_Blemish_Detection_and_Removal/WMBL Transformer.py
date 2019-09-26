import fastai
from fastai.vision import *
from fastai.callbacks import *
from PIL import Image, ImageDraw, ImageFont
import random as rand
from pathlib import Path
# Inspired from: https://tinyurl.com/y59gyjc7

class oldMold(object):

    def __init__(self, path_lr, path_hr, path_filter):

        """
        :type path_lr: str
        :type path_hr: str
        :type path_filter: str
        """

        # Passing in Low Res, High Res, and Filter Dir paths
        self.path_lr = path_lr
        self.path_hr = path_hr
        self.path_filter = path_filter

    def __call__(self, filename, i):

        """
        :type filename: str
        :type i: int
        """

        # Create dir if needed and load in images to Resize and Process
        dest = self.path_lr / filename.relative_to(self.path_hr)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(filename)
        targ_sz = resize_to(img, 300, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BICUBIC)

        # Randomly choose a Damage Augment and Overlay
        total_layers = os.listdir(self.path_filter)
        dmg_num = len(total_layers)
        damage = ('DC' + str(rand.randint(1, dmg_num)) + '.png')
        dmg_name = self.path_filter / damage
        angle = 90 * rand.randint(0, 3)
        top_layer = Image.open(dmg_name).rotate(angle)
        bottom_layer = img

        # Save file into Damage Subset
        bottom_layer.paste(top_layer, (0, 0), top_layer)
        bottom_layer.save(dest, "PNG")
