import streamlit as st
import random as rand
import os
import PIL.Image
import PIL.ImageFilter
from pathlib import Path
from fastai.vision import ImageImageList, ImageList
from model_utils import get_data, create_gen_learner

# Loading Paths for Model Load
path = Path('') # Path to data folder to load your model
path_hr = path / '' # Path to undamaged files
path_lr = path / '' # Path to damaged files

# Loading Paths to Inference
path_t = Path('') # Path to undamaged files
dmgpath = Path('') # Path to damage templates
inf_path = Path('') # Directory for inferencing

# Creating gen and Loading saved Weights
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
data_gen = get_data(1, 500, src, path_lr)
learn_gen = create_gen_learner(data_gen).load('') # LOAD MODEL HERE
test_list = ImageList.from_folder(inf_path)

# Starting Streamlit App with headline
st.markdown('**ML for Photo Repair**')
names = []
for filename in (os.listdir(path_t)):
    if '.png' in filename:
        names.append(filename)
    elif '.jpg' in filename:
        names.append(filename)
dmgnames = []
for filename in (os.listdir(dmgpath)):
    if '.png' in filename:
        dmgnames.append(filename)
        
option = st.selectbox('Select an Image:', (names))
dmgopt = st.selectbox('Select base Damage:', (dmgnames))
image = PIL.Image.open(path_t/option)
dmgimg = PIL.Image.open(dmgpath/dmgopt)
imlist = [image, dmgimg]
st.image(imlist, width=300)

if st.button('Generate Damaged Photo'):

    filelist = [ f for f in os.listdir(inf_path) if f.endswith(".png") or f.endswith('.jpg') ]
    for f in filelist:
        os.remove(os.path.join(inf_path, f))
    
    # Calling defs to create damage to a single image upon button press
    def vec_len(vec):
        return (vec[0] ** 2 + vec[1] ** 2) ** 0.5
    def pt_dist(pt1, pt2):
        return vec_len((pt1[0] - pt2[0], pt1[1] - pt2[1]))
    def clamp(val, minimum, maximum):
        return max(min(val, maximum), minimum)
    def warp(image, points):
        w_img = PIL.Image.new('RGBA', image.size, (255, 0, 0, 0))
        image_pixels = image.load()
        w_img_pixels = w_img.load()

        for y in range(image.size[1]):
            for x in range(image.size[0]):
                offset = [0, 0]
                for point in points:
                    point_position = (point[0] + point[2], point[1] + point[3])
                    shift_vector = (point[2], point[3])
                    helper = 1.5 / (3 * (pt_dist((x, y), point_position) / vec_len(shift_vector)) ** 4 + 1)
                    offset[0] -= helper * shift_vector[0]
                    offset[1] -= helper * shift_vector[1]
                coords = (
                clamp(x + int(offset[0]), 0, image.size[0] - 1), clamp(y + int(offset[1]), 0, image.size[1] - 1))
                w_img_pixels[x, y] = image_pixels[coords[0], coords[1]]
        return w_img

    def dmgSingle(im1_path, im2_path):
        # Randomly rotate damage layer before application
        angle = 90 * rand.randint(0, 3)
        imageN = PIL.Image.open(im2_path).rotate(angle)
        # Adding Random Warping
        p1 = (rand.randint(-200, -50), rand.randint(50, 200), rand.randint(-200, -50), rand.randint(50, 200))
        p2 = (rand.randint(-200, -50), rand.randint(50, 200), rand.randint(-200, -50), rand.randint(50, 200))
        imageN = warp(imageN, [p1, p2])       
        # Combining the images into one final Image
        background = PIL.Image.open(im1_path)
        background = background.resize((500, 500), resample=PIL.Image.BILINEAR).convert('RGB')
        background.paste(imageN, (0, 0), imageN)
        background.save(inf_path/'selected.png', "PNG")

    dmgSingle(path_t/option, dmgpath/dmgopt)
    selimg = PIL.Image.open(inf_path/'selected.png')
    st.image(selimg, caption='Ready to Inference', width=300)

# Calling inference on the given file
if st.button('Inference'):
    infimg = learn_gen.predict(test_list[0])[0]
    infimg.save(inf_path/'inf.png')
    
    userdmg = PIL.Image.open(inf_path/'selected.png')
    userinf = PIL.Image.open(inf_path/'inf.png')
    names_inf = [userdmg, userinf]
    
    st.image(names_inf, width=300)
else:
    pass
