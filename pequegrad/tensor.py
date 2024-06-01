from pequegrad.backend.c import Tensor
import pequegrad.backend.c as pg
import numpy as np
from typing import Union, List
from .utils import bind_method, bind_method_property
from .ops import (
    one_hot,
    cross_entropy_loss_indices,
    cross_entropy_loss_probs,
    log_softmax,
    logsumexp,
    softmax,
    conv2d,
    conv_transpose2d,
    max_pool2d,
    avg_pool2d,
    pad_constant,
    local_response_norm,
    dropout,
    transpose,
    var,
    std,
    sqrt,
    mse_loss,
    tensordot,
    sigmoid,
    tanh,
    erf,
    gelu,
    silu,
)


CUDA_AVAILABLE = True

_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]


bind_method(
    Tensor,
    "zeros",
    classmethod(
        lambda cls, shape, **kwargs: cls(np.zeros(shape).astype(np.float32), **kwargs)
    ),
)
bind_method(
    Tensor,
    "ones",
    classmethod(lambda cls, shape, **kwargs: cls(np.ones(shape), **kwargs)),
)
bind_method(Tensor, "relu", lambda x: pg.max(x, Tensor(0, device=x.device)))
bind_method(Tensor, "unfold", lambda *args, **kwargs: pg.im2col(*args, **kwargs))
bind_method(Tensor, "fold", lambda *args, **kwargs: pg.col2im(*args, **kwargs))
bind_method(Tensor, "one_hot", classmethod(one_hot))
bind_method(Tensor, "cross_entropy_loss_indices", cross_entropy_loss_indices)
bind_method_property(Tensor, "dim", lambda self: len(self.shape))
bind_method(Tensor, "cross_entropy_loss_probs", cross_entropy_loss_probs)
bind_method(Tensor, "log_softmax", log_softmax)
bind_method(Tensor, "logsumexp", logsumexp)
bind_method(Tensor, "softmax", softmax)
bind_method(
    Tensor,
    "max",
    lambda self, dim=None, keepdim=False: pg.max_reduce(self, dim, keepdim),
)
bind_method(Tensor, "exp", lambda self: pg.exp(self))
bind_method(
    Tensor, "sum", lambda self, dim=None, keepdim=False: pg.sum(self, dim, keepdim)
)
bind_method(Tensor, "log", lambda self: pg.log(self))
bind_method(
    Tensor, "mean", lambda self, dim=None, keepdim=False: pg.mean(self, dim, keepdim)
)
bind_method(
    Tensor,
    "reshape",
    lambda self, shape: pg.reshape(self, shape),
)
bind_method(Tensor, "conv_transpose2d", conv_transpose2d)
bind_method(Tensor, "conv2d", conv2d)
bind_method(Tensor, "max_pool2d", max_pool2d)
bind_method(Tensor, "avg_pool2d", avg_pool2d)
bind_method(Tensor, "transpose", lambda self, dim0, dim1: transpose(self, dim0, dim1))
bind_method_property(Tensor, "T", lambda self: transpose(self, 0, 1))
bind_method(Tensor, "__len__", lambda self: self.shape[0] if self.dim > 0 else 0)
bind_method(Tensor, "var", var)
bind_method(Tensor, "std", std)
bind_method(Tensor, "sqrt", sqrt)
bind_method(Tensor, "max_pool2d", max_pool2d)
bind_method(Tensor, "avg_pool2d", avg_pool2d)
bind_method(Tensor, "pad_constant", pad_constant)
bind_method(Tensor, "local_response_norm", local_response_norm)
bind_method(Tensor, "dropout", dropout)
bind_method(Tensor, "permute", lambda self, *dims: pg.permute(self, dims))
bind_method(Tensor, "mse_loss", mse_loss)
bind_method(Tensor, "tensordot", tensordot)
bind_method(Tensor, "sigmoid", sigmoid)
bind_method(Tensor, "tanh", tanh)
bind_method(pg, "sigmoid", sigmoid)
bind_method(pg, "tanh", tanh)
bind_method(Tensor, "erf", erf)
bind_method(Tensor, "gelu", gelu)
bind_method(Tensor, "silu", silu)
