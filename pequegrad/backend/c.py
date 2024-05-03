import os
import sys
import numpy as np
from typing import Optional
from typing import Tuple, Union
from typing import List


def bind_method(cls, existing, new):
    setattr(cls, existing, new)


def bind_method_property(cls, existing, new):
    setattr(cls, existing, property(new))


_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]


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
    classmethod(
        lambda cls, shape, **kwargs: cls(np.zeros(shape).astype(np.float32), **kwargs)
    ),
)
bind_method(
    Tensor,
    "ones",
    classmethod(lambda cls, shape, **kwargs: cls(np.ones(shape), **kwargs)),
)
bind_method(Tensor, "relu", lambda x: max(x, Tensor(0, device=x.device)))
bind_method(Tensor, "unfold", lambda *args, **kwargs: pg.im2col(*args, **kwargs))
bind_method(Tensor, "fold", lambda *args, **kwargs: pg.col2im(*args, **kwargs))


def one_hot(
    cls,
    num_classes: int,
    indices: "Tensor",
    requires_grad=False,
    device=device.cpu,
    dtype=dt.float32,
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
    dtypetonp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}
    np_one_hot = np.zeros((indices.shape[0], num_classes)).astype(dtypetonp[dtype])

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

    one_hot_target = Tensor.one_hot(
        self.shape[1], target, device=self.device, dtype=self.dtype
    )
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
bind_method(
    Tensor,
    "reshape",
    lambda self, shape: pg.reshape(self, shape),
)


def conv2d(
    self,
    filter: "Tensor",
    bias: "Tensor" = None,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
) -> "Tensor":
    """Returns the 2d convolution of the tensor with the given filter"""
    s_y, s_x = (stride, stride) if isinstance(stride, int) else stride
    d_y, d_x = (dilation, dilation) if isinstance(dilation, int) else dilation
    p_y, p_x = (padding, padding) if isinstance(padding, int) else padding

    # tensor is always of shape (batch, channels, height, width)
    # filter is always of shape (out_channels, in_channels, height, width)
    assert self.dim == 4, "conv2d is only supported for tensors with 4 dimensions"
    assert filter.dim == 4, "conv2d is only supported for filters with 4 dimensions"

    if p_y > 0 or p_x > 0:
        self = self.pad_constant((p_y, p_x, p_y, p_x))

    inp_unf = self.unfold(filter.shape[-2:], stride=(s_y, s_x), dilation=(d_y, d_x))
    out_unf = (
        inp_unf.transpose(1, 2) @ filter.reshape((filter.shape[0], -1)).T
    ).transpose(1, 2)
    after_conv_size = (
        (self.shape[-2] - (filter.shape[-2] - 1) * d_y - 1) // s_y + 1,
        (self.shape[-1] - (filter.shape[-1] - 1) * d_x - 1) // s_x + 1,
    )

    out = out_unf.fold(
        kernel_shape=(1, 1), output_shape=after_conv_size
    )  # dilation and strides are implicitly 1

    if bias is not None:
        assert (
            bias.shape[0] == out.shape[1]
        ), "bias shape must match output shape. Got {} but expected {}".format(
            bias.shape, (out.shape)
        )
        out += bias.reshape((1, -1, 1, 1))  # so we add the bias to each channel
    return out


def max_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d maxpooling of the tensor with the given kernel size"""
    assert self.dim == 4, "max_pool2d is only supported for tensors with 4 dimensions"
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (
        kernel_size
        if stride is None
        else (stride, stride)
        if isinstance(stride, int)
        else stride
    )
    assert stride[0] == stride[1], "kernel size must be square"

    new_shape = (
        self.shape[0],
        self.shape[1],
        (self.shape[2] - kernel_size[0]) // stride[0] + 1,
        (self.shape[3] - kernel_size[1]) // stride[1] + 1,
    )
    unfolded = self.unfold(kernel_size, stride=stride)
    # if there are multiple channels, each column of the unfolded tensor will be a flattened version of the
    # concatenation of the channels, so we need to reshape to 'divide' the channels
    unfolded = unfolded.reshape(
        (
            unfolded.shape[0],
            self.shape[1],
            kernel_size[0] * kernel_size[1],
            unfolded.shape[-1],
        )
    )
    maxed = unfolded.max(2)
    return maxed.reshape(new_shape)


def avg_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d average pooling of the tensor with the given kernel size"""
    assert self.dim == 4, "avg_pool2d is only supported for tensors with 4 dimensions"
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (
        kernel_size
        if stride is None
        else (stride, stride)
        if isinstance(stride, int)
        else stride
    )
    assert stride[0] == stride[1], "kernel size must be square"

    new_shape = (
        self.shape[0],
        self.shape[1],
        (self.shape[2] - kernel_size[0]) // stride[0] + 1,
        (self.shape[3] - kernel_size[1]) // stride[1] + 1,
    )
    unfolded = self.unfold(kernel_size, stride=stride)
    # if there are multiple channels, each column of the unfolded tensor will be a flattened version of the
    # concatenation of the channels, so we need to reshape to 'divide' the channels
    unfolded = unfolded.reshape(
        (
            unfolded.shape[0],  # batch
            self.shape[1],  # channels
            kernel_size[0] * kernel_size[1],  # kernel size 1 * kernel size 2
            unfolded.shape[-1],  # unfolded shape ('number of windows')
        )
    )
    maxed = unfolded.mean(2)
    return maxed.reshape(
        new_shape
    )  # once we have the mean, we reshape to the new shape


bind_method(Tensor, "conv2d", conv2d)
bind_method(Tensor, "max_pool2d", max_pool2d)
bind_method(Tensor, "avg_pool2d", avg_pool2d)


def transpose(self, dim0, dim1):
    axes = list(range(self.dim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return pg.permute(self, axes)


bind_method(Tensor, "transpose", lambda self, dim0, dim1: transpose(self, dim0, dim1))
bind_method_property(Tensor, "T", lambda self: transpose(self, 0, 1))
bind_method(Tensor, "__len__", lambda self: self.shape[0])


def var(self, dim=None, keepdim=True, correction=1):
    # keep dim on mean so that we can broadcast
    mean = self.mean(dim=dim, keepdim=True)

    # only dim of type positive int, tuple or None are supported
    assert (
        dim >= 0
        if isinstance(dim, int)
        else all(d >= 0 for d in dim)
        if dim is not None
        else True
    ), "only positive dims supported by now. Got {}".format(dim)

    N = (
        np.prod(self.shape)
        if dim is None
        else (
            self.shape[dim]
            if isinstance(dim, int)
            else np.prod([self.shape[i] for i in dim])
        )
    )
    variance = ((self - mean) ** 2).sum(dim=dim, keepdim=keepdim) / (N - correction)
    return variance


def std(self, dim=None, keepdim=True, correction=1):
    return self.var(dim=dim, keepdim=keepdim, correction=correction) ** 0.5


def sqrt(self):
    return self**0.5


def layer_norm(self, normalized_shape: _Shape, eps=1e-05):
    """Applies Layer Normalization over a mini-batch of inputs"""

    # calculate mean/std over last dims
    ns_l = len(normalized_shape)
    assert (
        self.dim >= ns_l
    ), "normalized_shape should be smaller than the number of dimensions of input tensor"
    assert list(self.shape[-ns_l:]) == list(
        normalized_shape
    ), "normalized_shape should be the last dimensions of input tensor"

    last_d_dims = tuple(range(self.dim - ns_l, self.dim))

    mean = self.mean(dim=last_d_dims, keepdim=True)
    variance = self.var(
        dim=last_d_dims, keepdim=True, correction=0
    )  # unbiased variance is used
    # for numerical stability, we add eps before sqrt
    std = (variance + eps).sqrt()

    return (self - mean) / std


bind_method(Tensor, "var", var)
bind_method(Tensor, "std", std)
bind_method(Tensor, "sqrt", sqrt)


def max_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d maxpooling of the tensor with the given kernel size"""
    assert self.dim == 4, "max_pool2d is only supported for tensors with 4 dimensions"
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (
        kernel_size
        if stride is None
        else (stride, stride)
        if isinstance(stride, int)
        else stride
    )
    assert stride[0] == stride[1], "kernel size must be square"

    new_shape = (
        self.shape[0],
        self.shape[1],
        (self.shape[2] - kernel_size[0]) // stride[0] + 1,
        (self.shape[3] - kernel_size[1]) // stride[1] + 1,
    )
    unfolded = self.unfold(kernel_size, stride=stride)
    # if there are multiple channels, each column of the unfolded tensor will be a flattened version of the
    # concatenation of the channels, so we need to reshape to 'divide' the channels
    unfolded = unfolded.reshape(
        (
            unfolded.shape[0],
            self.shape[1],
            kernel_size[0] * kernel_size[1],
            unfolded.shape[-1],
        )
    )
    maxed = unfolded.max(2)
    return maxed.reshape(new_shape)


def avg_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d average pooling of the tensor with the given kernel size"""
    assert self.dim == 4, "avg_pool2d is only supported for tensors with 4 dimensions"
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (
        kernel_size
        if stride is None
        else (stride, stride)
        if isinstance(stride, int)
        else stride
    )
    assert stride[0] == stride[1], "kernel size must be square"

    new_shape = (
        self.shape[0],
        self.shape[1],
        (self.shape[2] - kernel_size[0]) // stride[0] + 1,
        (self.shape[3] - kernel_size[1]) // stride[1] + 1,
    )
    unfolded = self.unfold(kernel_size, stride=stride)
    # if there are multiple channels, each column of the unfolded tensor will be a flattened version of the
    # concatenation of the channels, so we need to reshape to 'divide' the channels
    unfolded = unfolded.reshape(
        (
            unfolded.shape[0],  # batch
            self.shape[1],  # channels
            kernel_size[0] * kernel_size[1],  # kernel size 1 * kernel size 2
            unfolded.shape[-1],  # unfolded shape ('number of windows')
        )
    )
    maxed = unfolded.mean(2)
    return maxed.reshape(
        new_shape
    )  # once we have the mean, we reshape to the new shape


bind_method(Tensor, "max_pool2d", max_pool2d)
bind_method(Tensor, "avg_pool2d", avg_pool2d)
