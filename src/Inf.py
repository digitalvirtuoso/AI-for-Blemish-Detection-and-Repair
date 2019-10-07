import fastai
from fastai.vision import ImageList, ImageImageList
from pathlib import Path
from model_utils import get_data
from model_utils import create_gen_learner

path = Path('../data/')
path_hr = path / 'preprocessed'
path_lr = path / 'processed'
path_test = path / 'test_imgs'

# Gather and select Data / Output size
bs, size = 1, 500

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
data_gen = get_data(bs, size, src, path_hr)

# Load model to inference from
learn_gen = create_gen_learner(data_gen).load('') # Input model to load

# Open file to be inferenced
test_list = ImageList.from_folder(path_test)
test_list.open(test_list.items[0])

# Inference and display output
test_list[0].show(figsize=(7, 7),y=learn_gen.predict(test_list[0])[0])

# Save File if desired
#y = learn_gen.predict(test_list[8])[0]
#y.save(path_test/'inf1.png')
