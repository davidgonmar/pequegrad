import os
import sys
import numpy as np
from typing import Optional


def bind_method(cls, existing, new):
    setattr(cls, existing, new)


def bind_method_property(cls, existing, new):
    setattr(cls, existing, property(new))


build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise ImportError("Build path not found, please run `make` in the root directory")

from pequegrad_c import *  # noqa
import pequegrad_c as pg

bind_method(
    Tensor,
    "zeros",
    classmethod(lambda cls, shape, **kwargs: cls(np.zeros(shape), **kwargs)),
)
bind_method(
    Tensor,
    "ones",
    classmethod(lambda cls, shape, **kwargs: cls(np.ones(shape), **kwargs)),
)
bind_method(Tensor, "relu", lambda x: max(x, Tensor(0, device=x.device)))


def one_hot(
    cls, num_classes: int, indices: "Tensor", requires_grad=False, device=device.cpu
) -> "Tensor":
    indices = indices.numpy().astype(int)
    assert indices.ndim == 1, "indices must be a vector"
    assert np.all(
        indices >= 0
    ), "indices must be positive integers (>= 0), got {}".format(indices)
    assert np.all(
        indices < num_classes
    ), "indices must be smaller than num_classes, got {}".format(
        list(filter(lambda x: x >= num_classes, indices))
    )

    np_one_hot = np.zeros((indices.shape[0], num_classes))

    np_one_hot[np.arange(indices.shape[0]), indices] = 1.0

    return Tensor(np_one_hot, requires_grad=requires_grad, device=device)


bind_method(Tensor, "one_hot", classmethod(one_hot))


def cross_entropy_loss_indices(self, target: Tensor) -> Tensor:
    """
    Returns the cross entropy loss of the tensor.
    Only works for inputs of shape (batch, C), and targets of shape (batch,)
    """

    assert self.dim == 2, "input must be a matrix, of shape (batch, C)"
    assert target.dim == 1, "target must be a vector, of shape (batch,)"
    assert (
        self.shape[0] == target.shape[0]
    ), "input and target must have the same batch size, got {} and {}".format(
        self.shape, target.shape
    )

    one_hot_target = Tensor.one_hot(self.shape[1], target, device=self.device)

    return self.cross_entropy_loss_probs(one_hot_target)


bind_method(Tensor, "cross_entropy_loss_indices", cross_entropy_loss_indices)
bind_method_property(Tensor, "dim", lambda self: len(self.shape))


def log_softmax(self, dim=-1) -> "Tensor":
    """Returns the log softmax of the tensor"""
    # Use the logsumexp trick to avoid numerical instability
    return self - self.logsumexp(dim=dim, keepdim=True)


def cross_entropy_loss_probs(self, target: "Tensor") -> "Tensor":
    """Returns the cross entropy loss of the tensor"""

    assert self.shape == target.shape, "input and target must have the same shape"
    assert self.dim > 0, "input must be a vector"
    assert target.dim > 0, "target must be a vector"
    # At the moment, we expect (batch, C) tensors, both for input and target (probability distributions)
    # If there is no minibatch, we expect (C,) tensors
    # If there is a minibatch, we expect (batch, C, dim1, dim2, ...) tensors

    # We sum over the classes.
    # In case there is no minibatch, we'll sum over dim 0, which is the classes dimension, and mean
    # will not really do anything
    # If there is a minibatch, we'll sum over dim 1, which is the classes dimension, and reduce the minibatch
    # by taking the mean
    c_idx = 0 if self.dim == 1 else 1
    a = -(target * self.log_softmax(dim=c_idx)).sum(c_idx).mean()

    return a


bind_method(Tensor, "cross_entropy_loss_probs", cross_entropy_loss_probs)
bind_method(Tensor, "log_softmax", log_softmax)


def logsumexp(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
    """Returns the logsumexp of the tensor"""
    max = self.max(dim=dim, keepdim=True)
    return (self - max).exp().sum(dim=dim, keepdim=keepdim).log() + max


def softmax(self, dim=-1) -> "Tensor":
    """Returns the softmax of the tensor"""
    self_max = self.max(dim=dim, keepdim=True)

    softmax = (self - self_max).exp() / (self - self_max).exp().sum(
        dim=dim, keepdim=True
    )

    return softmax


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
