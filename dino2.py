#!/bin/env python
from sys import argv
from os import getenv, path

from sklearn.decomposition import PCA
from PIL.Image import open, Image, fromarray, Resampling

from gg import *
import gg


def remb(emb: np.ndarray, h: int, w: int) -> np.ndarray:
    m = int((len(emb) - 1) ** .5)
    ast(len(emb), 1 + m * m)
    return np.concat([emb[:1], np.stack([fromarray(i).resize((w, h), Resampling.BICUBIC) for i in emb[1:].reshape(m, m, -1).transpose(2, 0, 1)], 2).reshape(-1, emb.shape[1])])


def init(shape=(1, 126, 224, 3), m='/home/z/gguf/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf') -> None:
    global SHAPE, X, EMB, Y, PATCHSZ
    gg.init(m)
    SHAPE = shape
    PATCHSZ = getk(gg.GGUFCTX, 'patch_size')
    X = Tensor([shape[2], shape[1], shape[3], shape[0]])
    EMB = Tensor((gg.HIDDENSZ, 1 + shape[1]//PATCHSZ * shape[2]//PATCHSZ))
    Y = X.conv(Tensor('embeddings.patch_embeddings.projection.weight')).add_(Tensor('embeddings.patch_embeddings.projection.bias')).flatten(0, 1).T().cont().rcat(Tensor('embeddings.cls_token'), 1).add_(EMB).cat(Tensor('embeddings.register_tokens'), 1)
    for i in range(gg.LAYER):
        Y = Y.norm(f'encoder.layer.{i}.norm1').at(f'encoder.layer.{i}.attention').mul_(Tensor(f'encoder.layer.{i}.layer_scale1.lambda1')).add_(Y)
        Y = (Tensor.mlp if gg.LAYER != 40 else Tensor.swiglu)(Y.norm(f'encoder.layer.{i}.norm2'), f'encoder.layer.{i}.mlp').mul_(Tensor(f'encoder.layer.{i}.layer_scale2.lambda1')).add_(Y)
    ggml_build_forward_expand(gg.GF, Y.base)
    ggml_gallocr_alloc_graph(gg.ALLOC, gg.GF)
    EMB.fromnp(remb(Tensor('embeddings.position_embeddings').tonp()[0, 0], shape[1]//PATCHSZ, shape[2]//PATCHSZ))


def f(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not gg.GF:
        init(x.shape)
    ast(x.shape, SHAPE)
    X.fromnp((x-np.array([123.675, 116.28, 103.53], 'f')).transpose(0, 3, 1, 2)/np.array([58.395, 57.12, 57.375], 'f')[:, None, None])
    assert ggml_backend_graph_compute(gg.BACKEND, gg.GF) == GGML_STATUS_SUCCESS
    y = Y.tonp()
    return y[0, :, 0], y[0, :, 1:-4].reshape(SHAPE[0], SHAPE[1]//PATCHSZ, -1, gg.HIDDENSZ)


if __name__ == '__main__':
    argv = [i for i in argv[1:] if not path.exists(f'{i}dino2.{gg.HIDDENSZ}f')]
    x = [open(i) for i in argv]
    h = round(min(max(i.size[1] for i in x), 224)/14)*14
    w = round(min(max(i.size[0] for i in x), 224)/14)*14
    init((len(x), h, w, 3))
    x, z = f(np.stack([np.asarray(i.resize([w, h], Resampling.BOX))[..., :3] for i in x]))
    for i, x, Y in zip(argv, x, z):
        x.tofile(f'{i}dino2.{','.join(map(str, x.shape))}f')
        Y.tofile(f'{i}dino2.{','.join(map(str, Y.shape))}f')
    if getenv('PCA', ' '):
        z = PCA(3).fit_transform(z.reshape(-1, gg.HIDDENSZ)).reshape(*z.shape[:3], 3)
        z -= z.min((1, 2), keepdims=True)
        z *= 255.9 / z.max((1, 2), keepdims=True)
        for i, z in zip(argv, z.astype('B')):
            fromarray(z).resize((z.shape[1] * PATCHSZ, len(z) * PATCHSZ), Resampling.LANCZOS).save(f'{i}dino2.viz', 'jpeg')
