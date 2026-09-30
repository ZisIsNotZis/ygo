#!/bin/env python
from sklearn.decomposition import PCA
from PIL.Image import open, fromarray, BILINEAR, UnidentifiedImageError
from subprocess import call
from os import listdir
import numpy as np
imgs = {}
n = 1.2
for i in listdir():
    if i.split('.')[-1]not in ('jpeg', 'jpg', 'jp2', 'jxl', 'png', 'bmp', 'webp'):
        pass
    try:
        img = open(i)
    except UnidentifiedImageError as e:
        print(e)
        continue
    if 1/n-1e-3 <= img.size[0]/img.size[1] <= n+1e-3:
        continue
    imgs[i] = img.size
if imgs:
    call(['dinov2', *imgs])
for i, size in imgs.items():
    h = round(min(size[1], 336)/14)
    w = round(min(size[0], 336)/14)
    try:
        emb = np.asarray(fromarray(PCA(1).fit_transform(np.fromfile(f'{i}.{h},{w},384f', '384f')).reshape(w, -1)).resize(size, BILINEAR))
        if emb.mean()+emb[:len(emb)//2].mean() < np.concat([emb[::len(emb)-1].flat, emb[:, ::emb.shape[1]-1].flat]).mean()+emb[len(emb)//2:].mean():
            emb = -emb
        ax = np.argmax(size)
        open(i).crop((0, _ := ((_ := emb.sum(ax).cumsum())[int(emb.shape[ax]*n):]-_[:-int(emb.shape[ax]*n)]).argmax())[::ax*2-1]+(emb.shape[ax], _+emb.shape[ax]*n)[::ax*2-1]).convert('RGB').save(i)
    except (FileNotFoundError, OSError) as e:
        print(e)
