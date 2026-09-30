#!/bin/env python
from dino2 import init, f, fromarray, Resampling, np
from subprocess import Popen
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from os import getenv
from sys import argv


def getvid(A, y=4, x=8, I=range(9999), c=3):
    A = Popen(f'ffmpeg -hwaccel qsv -an -i {A} -vf vpp_qsv=framerate={R},scale_qsv=w={x}:h={y},hwdownload,format=nv12 -f rawvideo -pix_fmt {'rgb24'if c > 1 else 'gray'} -'.split(), stdout=-1).stdout  # ffmpeg -hwaccel qsv -an -c:v {ENC}_qsv -i {A} -vf vpp_qsv=framerate={R},scale_qsv=w={x}:h={y},hwdownload,format=nv12 -f rawvideo -pix_fmt {'rgb24'if c > 1 else 'gray'} -
    try:
        for i in range(I[-1]+1):
            a = A.read(y*x*c)
            if not a:
                break
            if i in I:
                yield np.frombuffer(a, ('B', (y, x, c)))[0]
    finally:
        A.close()


def consecutive(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.concat([[False], m, [False]])
    M = ~m
    return (M[:-1] & m[1:]).nonzero()[0].view('H')[::4], (m[:-1] & M[1:]).nonzero()[0].view('H')[::4]


def uniq(A, k=1700):
    A = np.linalg.norm(np.subtract(A[1:], A[:-1], dtype='f'), axis=1)/A.shape[1]**.5
    print(A)
    A, B = consecutive(A < k)
    print(A, B, sep='\n')
    A += B
    A >>= 1
    return A


def puniq(A, k=20):
    A, _ = pairwise_distances_argmin_min(KMeans(len(A)//k).fit(A).cluster_centers_, A)
    A.sort()
    return A


X0 = int(getenv('X0', 16))
Y0 = int(getenv('Y0', 16))
K0 = int(getenv('K0', 7))
X1 = int(getenv('X1', 504))
Y1 = int(getenv('Y1', 504))
K1 = int(getenv('K1', 1))
Y = int(getenv('Y', 0))
X = int(getenv('X', 0))
if not X or not Y:
    X, Y = eval(Popen(f'ffprobe -show_entries stream=width,height -of csv=p=0 {argv[1]}'.split(), stdout=-1).stdout.read())
ENC = getenv('ENC', 'av1')
R = getenv('R', 3)
VIS = getenv('VIS', '')
I = uniq(np.stack(list(getvid(argv[1], Y0, X0, c=1))).reshape(-1, Y0*X0), K0)
print(I)
J, A = f(np.stack(list(getvid(argv[1], Y1, X1, I))))
J = puniq(J, K1)
print(J)
I = I[J]
A = PCA(1).fit_transform(A[J].reshape(-1, 384)).reshape(-1, *A.shape[1:3])
A[A[:, :Y1//2].sum((1, 2)) < A[:, Y1//2:].sum((1, 2))] *= -1
A = np.stack([np.asarray(fromarray(a).resize((X, Y), Resampling.LANCZOS))for a in A])
A -= A.min((1, 2), keepdims=True)
A *= 255.9/A.max((1, 2), keepdims=True)
ax = int(Y < X)
J = A.sum(2-ax).cumsum(1)
J[:, min(X, Y):] -= J[:, :-min(X, Y)]
for i, j, a, b in zip(I, J[:, min(X, Y):].argmax(1), A.astype('B'), getvid(argv[1], Y, X, I)):
    if VIS:
        fromarray(a).save(f'{argv[1]}{i}.vis', 'jpeg')
    fromarray(b[*(slice(None),)*ax, j:j+min(X, Y)]).save(f'{argv[1]}{i}.webp')
