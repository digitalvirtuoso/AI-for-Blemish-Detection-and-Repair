import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
import os
from pathlib import Path
from wmtransformer import *
from model_utils import *

# Assigning default paths
path = Path('AI-for-Blemish-Detection-and-Repair/data/')
path_hr = path / 'preprocessed'
path_lr = path / 'processed'
path_filter = path / 'spot_dmg'

def accuracy_thresh_expand(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True)->Rank0Tensor:
    "Compute accuracy after expanding `y_true` to the size of `y_pred`."
    if sigmoid: y_pred = y_pred.sigmoid()
    return ((y_pred>thresh)==y_true[:,None].expand_as(y_pred).bool()).float().mean()

def create_dmg(path_hr, path_lr, newdmg = False):
    if len(os.listdir(path_lr)) < 0 or newdmg == True:
        itemlist = ImageList.from_folder(path_hr)
        parallel(oldMold(path_lr, path_hr, path_filter), itemlist.items)

# Create new augmented photos if necessary then load and split
create_dmg(path_hr, path_lr)
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)

bs, size = 64, 64

# Creating Databunch and Generative Learner
data_gen = get_data(bs, size, src, path_hr)
model_gen = create_gen_learner(data_gen)

# Pretraining and saving U-Net generator
model_gen.fit_one_cycle(2, pct_start=0.8) #2
model_gen.unfreeze()
model_gen.fit_one_cycle(10, slice(1e-6, 1e-3)) #10
# Use .show_results(rows=int) to view results

model_gen.save('gen-pre-cp1')

# Create Location for Predictions and save gen predictions
gen_file_loc = 'generated_images'
path_gen = path / gen_file_loc
path_gen.mkdir(exist_ok=True)
save_preds(model_gen, path_gen, data_gen.fix_dl)

model_gen = None
gc.collect()

# Create Databunch for Critic and create binary classifier Critic
data_crit = get_crit_data([gen_file_loc, 'preprocessed'], bs=bs, size=size, path=path)
model_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

# Pretraining and Saving Critic
model_critic.fit_one_cycle(6, 1e-3)
model_critic.save('critic-pre-cp1')

model_crit = None
model_gen = None
gc.collect()

#Assembling GAN
model_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre-cp1')
data_crit = get_crit_data(['damaged', 'preprocessed'], bs=bs, size=size, path=path)
model_gen = create_gen_learner(data_gen).load('gen-pre-cp1')
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(model_gen, model_crit, weights_gen=(1., 50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0., 0.99)), wd=1e-3)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
learn.fit(30, lr)
learn.save('gan-cp1')

learn.data = get_data(8, 256, src, path_hr)
learn.fit(10, lr / 2)
learn.save('gan-cp2')
# USE .show_results(rows=int) to view results from training