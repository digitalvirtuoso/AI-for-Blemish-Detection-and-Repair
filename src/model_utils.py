import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

def checkpt(ep, lr, model, interval, mod_name='Model_'):
    cpi = round(ep / interval)
    for z in range(cpi):
        model.fit(interval, lr)
        model.save(mod_name + str(z+1))

# The Following functions are from: https://tinyurl.com/yybfbbg4
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
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1

def create_gen_learner():
    return unet_learner(data_gen, arch, wd=1e-3, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=(-3., 3.), loss_func=MSELossFlat())

def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=1e-3)
