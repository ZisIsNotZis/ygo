from ctypes import POINTER, c_void_p, c_float, cast as ccast
from math import prod, ceil

from ggml import *
import numpy as np


def dbg[**P, T](f: Callable[P, T], l=[0]) -> Callable[P, T]:
    def dbg(*_: P.args, **__: P.kwargs) -> T:
        print('\t'*l[0]+f.__name__, *[_.shape if isinstance(_, Tensor) else _.shape if isinstance(_, np.ndarray) else _ for _ in _])
        l[0] += 1
        try:
            return f(*_)
        finally:
            l[0] -= 1
    return dbg


def ast[T](a: T, b: T) -> None:
    assert a == b, (a, b)


def getk(ctx_gguf: gguf_context_p, key: str) -> int:
    assert (t := gguf_find_key(ctx_gguf, key.encode())) >= 0, key
    return gguf_get_val_u32(ctx_gguf, t)


def get(ctx: ggml_context_p, tensor: str | bytes) -> ggml_tensor_p:
    assert (t := ggml_get_tensor(ctx, (tensor.encode() if isinstance(tensor, str) else tensor))), tensor
    return t


def gget(gf: ggml_cgraph_p, tensor: str | bytes) -> ggml_tensor_p:
    assert (t := ggml_graph_get_tensor(gf, (tensor.encode() if isinstance(tensor, str) else tensor))), tensor
    return t


class Tensor:
    def __init__(self, _: 'str | tuple[int,...] | ggml_tensor_p', name=''):
        self.base = get(CTX, _) if isinstance(_, str) else ggml_new_tensor_4d(GRAPHCTX, GGML_TYPE_F32, *_, *[1] * (4 - len(_))) if isinstance(_, (tuple, list)) else _
        if name:
            self.name(name)

    def name(self, _: str) -> 'Tensor':
        ggml_set_name(self.base, _.encode())
        return self

    @property
    def shape(self) -> list[int]:
        return list(self.base.contents.ne)

    @property
    def stride(self) -> list[int]:
        return list(self.base.contents.nb)

    @dbg
    def flatten(self, i=0, j=3, s=(-1,)) -> 'Tensor':
        ast(np.multiply(self.stride[i], self.shape[i:j]).tolist(), self.stride[i + 1:j + 1])
        shape = self.shape
        stride = self.stride
        if -1 in s:
            s = list(s)
            k = s.index(-1)
            s[k] = prod(shape[i:j + 1]) // prod(s[:k] + s[k + 1:])
        shape[i:j + 1] = s
        assert all(i == 1 for i in shape[4:])
        shape = shape[:4]
        shape += [1] * (4 - len(shape))
        stride[i:j + 1] = np.multiply(stride[i], [1, *s[:-1]])
        stride = stride[:4]
        stride += list(stride[-1:]) * (4 - len(stride))
        return Tensor(ggml_view_4d(GRAPHCTX, self.base, *shape, *stride[1:], 0))

    @dbg
    def permute(self, i=0, j=1, k=2, l=3) -> 'Tensor':
        return Tensor(ggml_permute(GRAPHCTX, self.base, i, j, k, l))

    def T(self, i=0, j=1) -> 'Tensor':
        dim = list(range(4))
        dim[i] = j
        dim[j] = i
        return self.permute(*dim)

    def tryexpand(self, _: 'Tensor', i=4) -> tuple[list[int], list[int]]:
        shape = [a if a > 1 or j == i else b for j, (a, b) in enumerate(zip(self.shape, _.shape))]
        shape_ = [a if a > 1 or j == i else b for j, (a, b) in enumerate(zip(_.shape, self.shape))]
        ast(shape[:i] + shape[i + 1:], shape_[:i] + shape_[i + 1:])
        return shape, shape_

    def expand(self, _: 'Tensor', i=4) -> tuple['Tensor', 'Tensor']:
        shape, shape_ = self.tryexpand(_, i)
        return self if shape == self.shape else Tensor(ggml_repeat(GRAPHCTX, self.base, Tensor(shape).base)), _ if shape_ == _.shape else Tensor(ggml_repeat(GRAPHCTX, _.base, Tensor(shape_).base))

    def cont(self) -> 'Tensor':
        return Tensor(ggml_cont(GRAPHCTX, self.base))

    def __getitem__(self, i: int | slice | tuple[int | slice]) -> 'Tensor':
        shape = self.shape
        stride = self.stride
        of = 0
        for j, i in enumerate(i if isinstance(i, tuple) else (i,)):
            i = i if isinstance(i, slice) else slice(i, i + 1)
            a = 0 if i.start is None else min(max(i.start + (i.start < 0) * shape[j], 0), shape[j])
            b = ceil((shape[j] if i.stop is None else min(max(i.stop + (i.stop < 0) * shape[j] - a, 0), shape[j])) / (i.step or 1))
            assert b > 0
            of += a * stride[j]
            shape[j] = b
            stride[j] *= i.step or 1
        return Tensor(ggml_view_4d(GRAPHCTX, self.base, *shape, *stride[1:], of))

    @dbg
    def cat(self, _: 'Tensor', i) -> 'Tensor':
        assert ggml_is_contiguous(self.base)
        assert ggml_is_contiguous(_.base)
        return Tensor(ggml_concat(GRAPHCTX, *[_.base for _ in self.expand(_, i)], i))

    @dbg
    def rcat(self, _: 'Tensor', i=1) -> 'Tensor':
        assert ggml_is_contiguous(self.base)
        assert ggml_is_contiguous(_.base)
        return Tensor(ggml_concat(GRAPHCTX, *[_.base for _ in _.expand(self, i)], i))

    def tonp(self) -> np.ndarray:
        ast(self.base.contents.type, GGML_TYPE_F32)
        return np.lib.stride_tricks.as_strided(np.asarray(ccast(self.base.contents.data, POINTER(c_float)).contents), self.shape[::-1], self.stride[::-1])

    def fromnp(self, x: np.ndarray) -> None:
        ast(self.base.contents.type, GGML_TYPE_F32)
        ast(self.shape[::-1], [1] * (4 - x.ndim) + list(x.shape))
        assert ggml_is_contiguous(self.base)
        x = np.ascontiguousarray(x)
        ggml_backend_tensor_set(self.base, x.ctypes.data, 0, x.nbytes)

    def gelu_(self) -> 'Tensor':
        return Tensor(ggml_gelu_inplace(GRAPHCTX, self.base))

    def silu_(self) -> 'Tensor':
        return Tensor(ggml_silu_inplace(GRAPHCTX, self.base))

    def softmax(self, scale) -> 'Tensor':
        return Tensor(ggml_soft_max_ext(GRAPHCTX, self.base, None, scale, 0.))

    @dbg
    def add_(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_)
        return Tensor(ggml_add_inplace(GRAPHCTX, self.base, _.base))

    @dbg
    def sub_(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_)
        return Tensor(ggml_sub_inplace(GRAPHCTX, self.base, _.base))

    @dbg
    def mul(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_)
        return Tensor(ggml_mul_inplace(GRAPHCTX, self.base, _.base))

    @dbg
    def mul_(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_)
        return Tensor(ggml_mul_inplace(GRAPHCTX, self.base, _.base))

    @dbg
    def dot(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_, 1)
        assert not ggml_is_transposed(self.base)
        assert not ggml_is_transposed(_.base)
        return Tensor(ggml_mul_mat(GRAPHCTX, _.base, self.base))

    @dbg
    def conv(self, _: 'Tensor') -> 'Tensor':
        ast(self.shape[2], _.shape[2])
        return Tensor(ggml_conv_2d_sk_p0(GRAPHCTX, _.base, self.base))

    def scale_(self, _: str) -> 'Tensor':
        return self.mul_(Tensor(_ + '.weight')).add_(Tensor(_ + '.bias'))

    def norm_(self, _: str) -> 'Tensor':
        return Tensor(ggml_norm_inplace(GRAPHCTX, self.base, 1e-6)).scale_(_)

    def norm(self, _: str) -> 'Tensor':
        return Tensor(ggml_norm(GRAPHCTX, self.base, 1e-6)).scale_(_)

    @dbg
    def lin(self, _: str) -> 'Tensor':
        return self.dot(Tensor(_ + '.weight')).add_(Tensor(_ + '.bias'))

    @dbg
    def mlp(self, _: str) -> 'Tensor':
        return self.lin(_ + '.fc1').gelu_().lin(_ + '.fc2')

    @dbg
    def swiglu(self, _: str) -> 'Tensor':
        x = self.lin(_ + '.weights_in')
        a = x[:x.shape[0] // 2].silu_()
        b = x[x.shape[0] // 2:]
        return a.mul_(b).lin('.weights_out')

    @dbg
    def at(self, _: str) -> 'Tensor':
        x = self.lin(_ + '.attention.qkv').flatten(0, 0, (-1, 3))  # cg 3 wh n
        q = x[:, 0].flatten(0, 1, (-1, HEAD)).T(1, 2)  # c wh0 g n
        k = x[:, 1].flatten(0, 1, (-1, HEAD)).T(1, 2)  # c wh1 g n
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5)  # wh1 wh0 g n
        v = x[:, 2].flatten(0, 1, (-1, HEAD)).permute(1, 2, 0).cont()  # wh1 c g n
        return qk.dot(v).T(1, 2).cont().flatten(0, 1).lin(_ + '.output.dense')  # cg 1 wh0 n

    def _rope(self, cos: 'Tensor', sin: 'Tensor') -> 'Tensor':
        a = self[::2]
        b = self[1::2]
        ab = a.mul(cos).sub_(b.mul(sin)).flatten(0, 1, (1, -1))
        ba = a.mul(sin).add_(b.mul(cos)).flatten(0, 1, (1, -1))
        return ab.cat(ba, 0).flatten(0, 1, self.shape[:2])

    def _rope2(self, cosh: 'Tensor', sinh: 'Tensor', cosw: 'Tensor', sinw: 'Tensor') -> 'Tensor':
        h = self[:self.shape[0] // 2]._rope(cosh, sinh)
        w = self[self.shape[0] // 2:]._rope(cosw, sinw)
        return h.cat(w, 0)

    def _atrope2(self, _: str, cosh: 'Tensor', sinh: 'Tensor', cosw: 'Tensor', sinw: 'Tensor', h: int) -> 'Tensor':
        x = self.lin(_ + '.attention.qkv').flatten(0, 1, (-1, 3))  # c g wh+5 n
        q = x[:, 0, :-5].flatten(1, 2, (-1, h))._rope2(cosh, sinh, cosw, sinw).flatten(1, 2, (1, -1)).cat(x[:, 0, -5:], 2).flatten(0, 1, (-1, HEAD))
        k = x[:, 1, :-5].flatten(1, 2, (-1, h))._rope2(cosh, sinh, cosw, sinw).flatten(1, 2, (1, -1)).cat(x[:, 1, -5:], 2).flatten(0, 1, (-1, HEAD))
        qk = q.dot(k).softmax(1 / q.shape[0] ** .5)  # wh1 wh0 g n
        v = x[:, 2].flatten(0, 1, (-1, HEAD)).permute(1, 2, 0).cont()  # wh1 c g n
        return qk.dot(v).T(1, 2).cont().flatten(0, 1).lin(_ + '.output.dense')  # cg 1 wh0 n

    def rope(self, _: 'Tensor') -> 'Tensor':
        self.tryexpand(_)
        return Tensor(ggml_rope_inplace(GRAPHCTX, self.base, _.base, self.shape[0], 4, imgsz//patc))
        a = self[::2]
        b = self[1::2]
        ab = a.mul(cos).sub_(b.mul(sin)).flatten(0, 1, (1, -1))
        ba = a.mul(sin).add_(b.mul(cos)).flatten(0, 1, (1, -1))
        return ab.cat(ba, 0).flatten(0, 1, self.shape[:2])


def init(model):
    global CTX, GGUFCTX, BACKEND, ALLOC, GRAPHCTX, HIDDENSZ, LAYER, HEAD, GF
    tmpctx = POINTER(ggml_context_p_ctypes)(c_void_p(0))
    ggufprm = gguf_init_params(False, tmpctx)
    GGUFCTX = gguf_init_from_file(model.encode(), ggufprm)
    prm = ggml_init_params(ggml_tensor_overhead() * (gguf_get_n_tensors(GGUFCTX) + 1), None, True)
    CTX = ggml_init(prm)
    for i in range(gguf_get_n_tensors(GGUFCTX)):
        name = gguf_get_tensor_name(GGUFCTX, i)
        src = get(tmpctx.contents, name)
        x = ggml_dup_tensor(CTX, src)
        ggml_set_name(x, name)
    BACKEND = ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(BACKEND, 28)
    ggml_backend_alloc_ctx_tensors(CTX, BACKEND)
    for i in range(gguf_get_n_tensors(GGUFCTX)):
        name = gguf_get_tensor_name(GGUFCTX, i)
        src = get(tmpctx.contents, name)
        x = get(CTX, name)
        ggml_backend_tensor_set(x, ggml_get_data(src), 0, ggml_nbytes(src))
    ALLOC = ggml_gallocr_new(ggml_backend_get_default_buffer_type(BACKEND))
    graphprm = ggml_init_params(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), None, True)
    GRAPHCTX = ggml_init(graphprm)
    GF = ggml_new_graph(GRAPHCTX)
    HIDDENSZ = getk(GGUFCTX, 'hidden_size')
    LAYER = getk(GGUFCTX, 'num_hidden_layers')
    HEAD = getk(GGUFCTX, 'num_attention_heads')


BACKEND: ggml_backend_t = None
GGUFCTX: gguf_context_p = None
CTX: ggml_context_p = None
GRAPHCTX: ggml_context_p = None
ALLOC: ggml_gallocr = None
GF: ggml_cgraph_p = None
HIDDENSZ = LAYER = HEAD = 0
