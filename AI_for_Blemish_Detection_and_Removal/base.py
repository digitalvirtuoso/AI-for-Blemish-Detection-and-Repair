import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import random as rand
from pathlib import Path
from wmtransformer import oldMold

def checkpt(ep, lr, model, interval, mod_name='Model_'):
    cpi = round(ep / interval)
    for z in range(cpi):
        model.fit(interval, lr)
        model.save(mod_name + str(z+1))

# The Following functions (and model is inspired) from: https://tinyurl.com/yybfbbg4
def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr / x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

def get_crit_data(classes, bs, size):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
            .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

def save_preds(dl):
    i = 0
    names = dl.dataset.items
    for b in dl:
        preds = model_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1

def create_gen_learner(pretrain=True):
    # Loading resnet34 into unet to pretrain
    if pretrain is True:
        arch = models.resnet34
        #arch()
    return unet_learner(data_gen, arch, wd=1e-3, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=(-3., 3.), loss_func=MSELossFlat())

def create_critic_learner(data, metrics):
    # Assigning loss function for Critic and creating the critic Model
    loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=1e-3)


# ASSIGNING PATHS - CREATING DIRs
path = Path('/data/')
path_hr = path / 'raw'
path_lr = path / 'processed'
path_filter = path / 'spot_dmg'

# Creating image list and processing images
itemlist = ImageList.from_folder(path_hr)
parallel(oldMold(path_lr, path_hr, path_filter), itemlist.items)

# Loading and splitting dataset into Train and Validation sets
src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)
bs, size = 64, 64

# Creating Databunch and Generative Learner
data_gen = get_data(bs, size)
model_gen = create_gen_learner()

# Pretraining and saving U-Net generator
model_gen.fit_one_cycle(2, pct_start=0.8)
model_gen.unfreeze()
model_gen.fit_one_cycle(6, slice(1e-6, 1e-3))
# Use .show_results(rows=int) to view results

# Saving Weights
model_gen.save('gen-pre1')

# Create Location for Predictions
gen_file_loc = 'generated_images'
path_gen = path / gen_file_loc
path_gen.mkdir(exist_ok=True)

# Saving Generator Predictions
save_preds(data_gen.fix_dl)
model_gen = None
gc.collect()

# Creating Databunch for Critic and creating Critic as a binary classifier
data_crit = get_crit_data([gen_file_loc, 'images'], bs=bs, size=size)
model_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

# Pretraining and Saving Critic
model_critic.fit_one_cycle(6, 1e-3)
model_critic.save('critic-pre1')
model_crit = None
model_gen = None
gc.collect()


#Assembling GAN
model_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre1')
data_crit = get_crit_data(['damaged', 'images'], bs=bs, size=size)
model_gen = create_gen_learner().load('gen-pre1')
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(model_gen, model_crit, weights_gen=(1., 50.), show_img=False, switcher=switcher,
                                 opt_func=partial(optim.Adam, betas=(0., 0.99)), wd=1e-3)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))


lr = 1e-4
learn.fit(30, lr)
learn.save('gan-checkpoint-cp1')

learn.data = get_data(8, 256)
learn.fit(10, lr / 2)
# USE .show_results(rows=3) to view results from training
learn.save('gan-checkpoint-cp2')
