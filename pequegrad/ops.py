import numpy as np
from typing import Optional, Tuple, Union, List
import pequegrad.backend.c as pg
import math
from pequegrad.backend.c import (
    custom_init as _custom_init,
    permute,
)

Tensor = pg.Tensor
dt = pg.dt
_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]
dtypetonp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}

from pequegrad.backend.c import Device, from_str  # noqa
from pequegrad import extend


class DeviceModule:
    @staticmethod
    def cuda(idx: int = 0) -> Device:
        return from_str(f"cuda:{idx}")

    @staticmethod
    def cpu(idx: int = 0) -> Device:
        return from_str(f"cpu:{idx}")

    @staticmethod
    def from_str(device: str) -> Device:
        if isinstance(device, Device):
            return device
        return from_str(device)


device = DeviceModule


def custom_init(f):
    def ff(*args, **kwargs):
        res = f(*args, **kwargs)
        if isinstance(res, tuple):
            return res
        if isinstance(res, Tensor):
            return (res,)
        else:
            raise ValueError("custom_init must return a Tensor or a tuple of Tensors")

    ff.__name__ = f.__name__  # so we don't lose the name of the function
    p = _custom_init(ff)

    return p


def as_contiguous(x: "Tensor") -> "Tensor":
    return pg.as_contiguous(x)


def tensordot(a: "Tensor", b: "Tensor", dims: Union[int, Tuple[List[int], List[int]]]):
    # Convert dims to a consistent format (it accepts both a single int or a tuple of lists of ints)
    if isinstance(dims, int):
        dims = (list(range(-dims, 0)), list(range(dims)))
    a_axes, b_axes = dims
    # make a_axes have positive values
    a_axes = [a.dim + ax if ax < 0 else ax for ax in a_axes]
    b_axes = [b.dim + ax if ax < 0 else ax for ax in b_axes]

    # Compute the new shape for a
    a_shape = list(a.shape)
    a_shape1 = [
        a_shape[i] for i in range(len(a_shape)) if i not in a_axes
    ]  # shape of a without the axes to sum over
    a_shape2 = [a_shape[i] for i in a_axes]  # shape of a to sum over
    a_reshape = [-1, int(np.prod(a_shape2))]

    # Compute the new shape for b
    b_shape = list(b.shape)
    b_shape1 = [
        b_shape[i] for i in range(len(b_shape)) if i not in b_axes
    ]  # shape of b without the axes to sum over
    b_shape2 = [b_shape[i] for i in b_axes]  # shape of b to sum over
    b_reshape = [int(np.prod(b_shape2)), -1]

    # Permute the dimensions of a and b
    a_perm = [
        i for i in range(len(a.shape)) if i not in a_axes
    ] + a_axes  # pull the axes to sum over to the end
    b_perm = b_axes + [
        i for i in range(len(b.shape)) if i not in b_axes
    ]  # pull the axes to sum over to the start

    # Flatten a and b to 2D tensors of shape (a_shape1, a_shape2) and (b_shape2, b_shape1)
    a_t = a.permute(*a_perm).reshape(a_reshape)
    b_t = b.permute(*b_perm).reshape(b_reshape)

    assert a_t.shape[1] == b_t.shape[0], "shapes {} and {} not aligned".format(
        a_t.shape, b_t.shape
    )
    result = pg.matmul(a_t, b_t)

    # Reshape the result back to the correct shape
    final_shape = a_shape1 + b_shape1
    return result.reshape(final_shape)


broadcast_to = pg.broadcast_to


def gelu(self, approximate: str = None):
    if not approximate:
        # raise NotImplementedError("gelu not implemented yet without approximation")
        approximate = "tanh"
    if approximate == "tanh":
        return (
            0.5
            * self
            * (1.0 + pg.tanh(math.sqrt(2 / math.pi) * (self + 0.044715 * (self**3.0))))
        )


def erf(x):
    raise NotImplementedError("erf not implemented yet")


def silu(self):
    return self * pg.sigmoid(self)


def dropout(self, p: float, training: bool = True) -> "Tensor":
    """Returns the dropout of the tensor"""
    if training:
        mask = pg.binomial(
            1 - p, self.shape, device=self.device, dtype=dt.float32
        ).astype(self.dtype)
        return self * mask / (1 - p)
    return self


def mse_loss(self, target: "Tensor") -> "Tensor":
    """Returns the mean squared error loss of the tensor"""
    return ((self - target) ** 2).mean()


pg.abs = lambda x: (x**2) ** 0.5  # TODO -- I am lazy
Tensor.abs = pg.abs
_abs = pg.abs
where = lambda condition, x, y: pg.where(condition, x, y)


def sigmoid(self):
    return 1 / (1 + pg.exp(-self))


def tanh(self):
    # compute it manually
    return (pg.exp(self) - pg.exp(-self)) / (pg.exp(self) + pg.exp(-self))


def safetanh(self):
    # 2 / (1 + torch.exp(-2 * x)) - 1
    return 2 / (1 + pg.exp(-2 * self)) - 1


tanh = safetanh

dtypetonp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}

_device = device


def one_hot(
    num_classes: int,
    indices: "Tensor",
    device="cpu",
    dtype=dt.float32,
) -> "Tensor":
    # if indices is an int, fast path
    if isinstance(indices, int):
        nparr = np.zeros(num_classes).astype(dtypetonp[dtype])
        nparr[indices] = 1.0
        return Tensor(nparr, device=device)
    assert indices.ndim in [1, 0], "indices must be a vector"

    if indices.device == _device.cpu(0):
        indices = indices.numpy()
        np_one_hot = np.zeros((indices.shape[0], num_classes)).astype(dtypetonp[dtype])

        np_one_hot[np.arange(indices.shape[0]), indices] = 1.0
        ret = Tensor(np_one_hot, device=device)
        return ret
    else:
        return pg.one_hot(indices, num_classes)


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

    """one_hot_target = Tensor.one_hot(
        self.shape[1], target, device=self.device, dtype=self.dtype
    )
    return self.cross_entropy_loss_probs(one_hot_target)"""

    # it is faster to take the indices of the target and use them to index the output of the softmax
    # than to one-hot encode the target and then multiply the one-hot encoding with the output of the softmax

    # we need to take the log of the softmax of the input
    log_softmax = self.log_softmax(dim=1)
    # we need to take the negative of the log softmax of the target
    return -(log_softmax[:, target].mean())


def log_softmax(self, dim=-1) -> "Tensor":
    """Returns the log softmax of the tensor"""
    # Use the logsumexp trick to avoid numerical instability
    return self - self.logsumexp(dim=dim, keepdim=True)


def cross_entropy_loss_probs(self, target: "Tensor") -> "Tensor":
    """Returns the cross entropy loss of the tensor"""

    assert (
        self.shape == target.shape
    ), "input and target must have the same shape, got {} and {}".format(
        self.shape, target.shape
    )
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


def logsumexp(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
    """Returns the logsumexp of the tensor"""
    m = self.max_reduce(dim=dim, keepdim=True)
    return (self - m).exp().sum(dim=dim, keepdim=keepdim).log() + m


def softmax(self, dim=-1) -> "Tensor":
    """Returns the softmax of the tensor"""
    self_max = self.max_reduce(dim=dim, keepdim=True)

    softmax = (self - self_max).exp() / (self - self_max).exp().sum(
        dim=dim, keepdim=True
    )

    return softmax


def _prepare_filter_for_conv_gemm(filter: "Tensor", groups: int) -> "Tensor":
    # filter of shape {out_channels, in_channels // groups, k_h, k_w}
    if groups == 1:
        return filter.reshape((filter.shape[0], -1)).permute(
            1, 0
        )  # shape {in_channels * k_h * k_w, out_channels}
    else:
        return filter.reshape((groups, filter.shape[0] // groups, -1)).permute(
            0, 2, 1
        )  # shape {groups, in_channels // groups * k_h * k_w, out_channels // groups}


def conv2d(
    self,
    filter: "Tensor",
    bias: "Tensor" = None,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
) -> "Tensor":
    """Returns the 2d convolution of the tensor with the given filter"""
    s_y, s_x = (stride, stride) if isinstance(stride, int) else stride
    d_y, d_x = (dilation, dilation) if isinstance(dilation, int) else dilation
    p_y, p_x = (padding, padding) if isinstance(padding, int) else padding

    # tensor is always of shape (batch, channels, height, width)
    # filter is always of shape (out_channels, in_channels // groups, k_h, k_w)
    assert self.dim == 4, "conv2d is only supported for tensors with 4 dimensions"
    assert filter.dim == 4, "conv2d is only supported for filters with 4 dimensions"

    if p_y > 0 or p_x > 0:
        self = self.pad_constant((p_y, p_x, p_y, p_x))

    inp_unf = self.unfold(
        filter.shape[-2:], stride=(s_y, s_x), dilation=(d_y, d_x)
    )  # shape {batch_size, in_channels * k_h * k_w, out_h * out_w}
    # now reshape for groups
    inp_unf = (
        inp_unf.reshape((inp_unf.shape[0], groups, -1, inp_unf.shape[-1]))
        if groups != 1
        else inp_unf
    )  # shape {batch_size, groups, in_channels // groups * k_h * k_w, out_h * out_w}

    # filter is of shape {out_channels, in_channels // groups, k_h, k_w}

    # {groups, out_channels // groups, k_h * k_w * in_channels // groups}) @ ({batch_size, groups, in_channels // groups * k_h * k_w, out_h * out_w})
    out_unf = (
        _prepare_filter_for_conv_gemm(filter, groups).transpose(-2, -1) @ inp_unf
    )  # shape {batch_size, groups, out_channels // groups, out_h * out_w}

    # merge groups
    out_unf = (
        out_unf.reshape((out_unf.shape[0], -1, out_unf.shape[-1]))
        if groups != 1
        else out_unf
    )  # shape {batch_size, out_channels, out_h * out_w}

    after_conv_size = (
        (self.shape[-2] - (filter.shape[-2] - 1) * d_y - 1) // s_y + 1,
        (self.shape[-1] - (filter.shape[-1] - 1) * d_x - 1) // s_x + 1,
    )

    # fold again
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


def conv_transpose2d(
    self,
    filter: "Tensor",
    bias: "Tensor" = None,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
) -> "Tensor":
    """Returns the 2d transposed convolution of the tensor with the given filter"""
    s_y, s_x = (stride, stride) if isinstance(stride, int) else stride
    p_y, p_x = (padding, padding) if isinstance(padding, int) else padding
    op_y, op_x = (
        (
            output_padding,
            output_padding,
        )
        if isinstance(output_padding, int)
        else output_padding
    )
    d_y, d_x = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert (
        self.dim == 4
    ), "conv_transpose2d is only supported for tensors with 4 dimensions"
    assert (
        filter.dim == 4
    ), "conv_transpose2d is only supported for filters with 4 dimensions"
    inp_unf = self.unfold(
        (1, 1), (1, 1), (1, 1)
    )  # shape (batch, channels, height, width) -> (batch, channels, 1, height * width)
    x = (inp_unf.transpose(1, 2) @ filter.reshape((filter.shape[0], -1))).transpose(
        1, 2
    )
    after_convt_size = (
        (self.shape[2] - 1) * s_y + (filter.shape[-2] - 1) * d_y + 1 + op_y,
        (self.shape[3] - 1) * s_x + (filter.shape[-1] - 1) * d_x + 1 + op_x,
    )
    out = x.fold(
        kernel_shape=filter.shape[-2:],
        output_shape=after_convt_size,
        stride=(s_y, s_x),
        dilation=(d_y, d_x),
    )
    # if padding is not 0, we need to crop the output
    if p_y > 0 or p_x > 0:
        out = out[:, :, p_y:-p_y, p_x:-p_x]

    if bias is not None:
        assert (
            bias.shape[0] == out.shape[1]
        ), "bias shape must match output shape (channels). Got {} but expected {}".format(
            bias.shape, (out.shape)
        )
        # bias of shape (out_channels,) -> (1, out_channels, 1, 1)
        bias = bias.reshape((1, -1, 1, 1))
        out = out + bias
    return out


def depthwise_separable_conv2d(
    self,
    filter: "Tensor",
    bias: "Tensor" = None,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
):
    ngroups = filter.shape[1]  # each channel is a group
    return conv2d(filter, bias, stride, dilation, padding, groups=ngroups)


def _pool2d(
    self,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = None,
    pool_type: str = "invalid",
) -> "Tensor":
    """Returns the 2d pooling of the tensor with the given kernel size and type"""
    assert self.dim == 4, "pool2d is only supported for tensors with 4 dimensions"
    kernel_size = (
        (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    )
    stride = (
        kernel_size
        if stride is None
        else (stride, stride) if isinstance(stride, int) else stride
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
    pooled = unfolded.__getattribute__(pool_type)(2)
    return pooled.reshape(new_shape)


def max_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d maxpooling of the tensor with the given kernel size"""
    return _pool2d(self, kernel_size, stride, "max_reduce")


def avg_pool2d(
    self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
) -> "Tensor":
    """Returns the 2d average pooling of the tensor with the given kernel size"""
    return _pool2d(self, kernel_size, stride, "mean")


def transpose(self, dim0, dim1):
    if self.dim == 1:
        return self
    axes = list(range(self.dim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return permute(self, axes)


def var(self, dim=None, keepdim=True, correction=1):
    # keep dim on mean so that we can broadcast
    mean = self.mean(dim=dim, keepdim=True)

    # only dim of type positive int, tuple or None are supported
    assert (
        dim >= 0
        if isinstance(dim, int)
        else all(d >= 0 for d in dim) if dim is not None else True
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


matmul = lambda a, b: a @ b


def layer_norm(self, normalized_shape: _Shape, eps=1e-05, weight=None, bias=None):
    """Applies Layer Normalization over a mini-batch of inputs"""

    # calculate mean/std over last dims
    normalized_shape = (
        list(normalized_shape)
        if isinstance(normalized_shape, tuple)
        else [normalized_shape]
    )
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

    # apply normalization
    x = (self - mean) / std

    # apply weight and bias
    if weight is not None:
        x = x * weight

    if bias is not None:
        x = x + bias

    return x


pg.layer_norm = layer_norm


def split(self, split_size: int, dim: int = 0) -> List["Tensor"]:
    """
    Splits the tensor into chunks of the given size along a given dimension
    """
    assert (
        self.shape[dim] % split_size == 0
    ), "tensor must be divisible by the split size"
    slices_for_getdim = [slice(None) for _ in range(self.dim)]
    splits = []
    for i in range(0, self.shape[dim], split_size):
        slices_for_getdim[dim] = slice(i, i + split_size)
        splits.append(self[tuple(slices_for_getdim)])
    return splits


def tril(self, diagonal: int = 0) -> "Tensor":
    """
    Returns the lower triangular part of the tensor
    """
    row_indices = arange(0, self.size(-2)).reshape((-1, 1)).to(self.device)
    col_indices = arange(0, self.size(-1)).reshape((1, -1)).to(self.device)
    mask = (row_indices >= (col_indices - diagonal)).astype(self.dtype)
    return self * mask


def triu(self, diagonal: int = 0) -> "Tensor":
    """
    Returns the upper triangular part of the tensor
    """
    row_indices = arange(0, self.size(-2)).reshape((-1, 1)).to(self.device)
    col_indices = arange(0, self.size(-1)).reshape((1, -1)).to(self.device)
    mask = (row_indices <= (col_indices - diagonal)).astype(self.dtype)
    return self * mask


def pad_constant(x: Tensor, pad: _Shape, constant: float = 0.0):
    pad = list(pad)  # for a 1d pad on last dim, it will be (padleft, padright)
    new_shape = list(x.shape)

    padpairs = list(
        zip(pad[::2], pad[1::2])
    )  # will split (a, b, c, d) into [(a, b), (c, d)]
    padpairs = list(reversed(padpairs))  # to match torch's behavior
    # which means "on last dim, pad a on left, b on right, and on last-1 dim, pad c on left, d on right"

    # pad padpairs with 0 for each dimension that is not being padded, to the start of the list
    for _ in range(len(new_shape) - len(padpairs)):
        padpairs.insert(0, (0, 0))

    # now we can calculate the new shape
    for i, (padleft, padright) in enumerate(padpairs):
        new_shape[i] += padleft + padright

    if x.device == device.cpu(0):
        out = pg.broadcast_to(
            pg.fill(
                (),
                x.dtype,
                constant,
                x.device,
            ),
            new_shape,
        )
    else:
        # TODO -- buggy on cuda at the moment
        out = pg.fill(
            new_shape,
            x.dtype,
            constant,
            x.device,
        )
    slices = [slice(int(pad[0]), int(-pad[1])) for pad in padpairs]

    for i, _slice in enumerate(slices):
        if _slice.start == 0 and _slice.stop == 0:
            slices[i] = slice(None, None, None)  # same as a[:]

    slices = tuple(slices)

    out = pg.assign_at(out, x, slices)

    return out


assign_at = pg.assign_at
fill = pg.fill


def zeros(shape: _Shape, dtype=dt.float32, dev="cpu"):
    return fill(shape, dtype, 0.0, device.from_str(dev))


@custom_init
def randn(shape: int, dtype=dt.float32, device="cpu"):
    return Tensor(np.random.randn(*shape).astype(dtypetonp[dtype]), device=device)


@custom_init
def rand(shape: int, dtype=dt.float32, device="cpu"):
    return Tensor(np.random.rand(*shape).astype(dtypetonp[dtype]), device=device).to(
        device
    )


def outer_prod(v1, v2):
    return v1.reshape((v1.shape[0], 1)) * v2.reshape((1, v2.shape[0]))


mean = lambda t, *args, **kwargs: t.mean(*args, **args)


def local_response_norm(
    self, size: int, alpha: float, beta: float, k: float
) -> "Tensor":
    """
    Returns the local response normalization of the tensor

    Args:
        size: The size of the normalization window
        alpha: Multiplicative factor
        beta: Exponent
        k: Additive factor
    """

    assert (
        self.dim == 4
    ), "local_response_norm is only supported for tensors with 4 dimensions, got {}".format(
        self
    )

    # input of shape (batch, in_channels, height, width)
    # we need to normalize accross channels

    # first, pad the input so that we can calculate the normalization for the borders
    # (so the normalized output has the same shape as the input)
    pad = (size - 1) // 2
    padded = self.pad_constant((pad, pad, pad, pad), constant=0)

    # shape self: (batch, in_channels, height, width) -> (batch, size * size * in_channels, height, width)
    unfolded = padded.unfold((size, size), stride=(1, 1)).reshape(
        (self.shape[0], -1, self.shape[2], self.shape[3])
    )

    # shape unfolded: (batch, size * size * in_channels, height, width) -> (batch, 1, height, width)
    norm_factor = (unfolded**2).mean(1, keepdim=True) * alpha + k

    return self / norm_factor**beta


def chunk(self, chunks: int, dim: int = 0) -> List["Tensor"]:
    """
    Splits the tensor into a number of chunks along a given dimension
    """
    assert (
        self.shape[dim] % chunks == 0
    ), "tensor must be divisible by the number of chunks"
    chunk_size = self.shape[dim] // chunks
    slices_for_getdim = [slice(None) for _ in range(self.dim)]
    chunks_ = []
    for ch in range(chunks):
        slices_for_getdim[dim] = slice(ch * chunk_size, (ch + 1) * chunk_size)
        chunks_.append(self[tuple(slices_for_getdim)])
    return chunks_


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Concatenates the given sequence of tensors in the given dimension
    """
    n = len(tensors)

    if n == 1:
        return tensors[0]
    assert all(
        tensors[0].shape[i] == t.shape[i]
        for t in tensors[1:]
        for i in range(len(tensors[0].shape))
        if i != dim
    ), "all input tensors must have the same shape in dimensions other than {}, got {}".format(
        dim, [t.shape for t in tensors]
    )

    new_shape = list(tensors[0].shape) if tensors[0].ndim > 0 else [0]

    def _sum_dim(tensors, dim):
        if tensors[0].ndim == 0:
            return len(tensors)
        res = 0
        for t in tensors:
            res += t.shape[dim]
        return res

    new_shape[dim] = _sum_dim(tensors, dim)

    out = pg.fill(
        new_shape,
        tensors[0].dtype,
        0,
        tensors[0].device,
    )
    start = 0
    for t in tensors:
        slices = [slice(None) for _ in range(_max(t.dim, 1))]
        slices[dim] = slice(start, start + t.shape[dim] if t.ndim > 0 else 1)
        out = pg.assign_at(out, t, tuple(slices))
        start += t.shape[dim] if t.ndim > 0 else 1

    return out


def pad_to(self, i: int, value: float = 0.0):
    """
    Pads a vector with value until it reaches the desired length
    """
    assert self.dim == 1, "pad_to is only supported for 1d tensors"
    # use pad constant only on the right
    # do it in numpy for now
    npt = self.numpy()
    if npt.shape[0] >= i:
        return self
    return Tensor(
        np.pad(npt, (0, i - npt.shape[0]), mode="constant", constant_values=value),
        device=self.device,
    )


def matmul_with_reshapes(self, other):
    assert self.ndim >= 2 and other.ndim >= 2
    a = self.unsqueeze(-1)  # [d1, d2, ..., m, k, 1]
    b = other.unsqueeze(-3)  # [d1, d2, ..., 1, k, n]
    pro = a * b
    x = pro.sum(-2)
    return x


def scaled_dot_product_attention(
    query: "Tensor",
    key: "Tensor",
    value: "Tensor",
    mask: "Tensor" = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> "Tensor":
    """
    Scaled dot-product attention mechanism
    """
    if is_causal:
        raise NotImplementedError("Causal attention not implemented yet, use a mask")
    d_k = key.shape[-1]
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = pg.where(
            mask,
            scores,
            pg.broadcast_to(
                pg.fill((), scores.dtype, -1e9, scores.device), scores.shape
            ),
        )
    p_attn = softmax(scores, dim=-1)
    if dropout_p > 0.0:
        p_attn = dropout(p_attn, dropout_p)
    return p_attn @ value


def truncate(self, i: int):
    """
    Truncates a floating point vector to the i-th fractional digit
    """
    return (self * 10**i).astype(dt.int32) / 10**i


"""pg.matmul = matmul_with_reshapes

Tensor.__matmul__ = matmul_with_reshapes"""


sum = pg.sum
log = pg.log
_max = max
max = pg.max
exp = pg.exp
gt = pg.gt
lt = pg.lt
mul = pg.mul
div = pg.div
pow = pg.pow
squeeze = pg.squeeze
unsqueeze = pg.unsqueeze
sub = pg.sub
add = pg.add
neq = pg.neq
im2col = pg.im2col
col2im = pg.col2im


def norm(vector: "Tensor", p: float = 2) -> "Tensor":
    """
    Returns the p-norm of the tensor
    """
    exped = pg.abs(vector) ** p
    return pg.sum(exped) ** (1 / p)


pg.norm = norm


def slerp(p0: Tensor, p1: Tensor, alpha: float) -> Tensor:
    assert p0.ndim == p1.ndim and p0.shape == p1.shape and p0.ndim == 1
    # computes Spherical Linear intERPolation between two points
    # described in https://www.cs.cmu.edu/~kiranb/animation/p245-shoemake.pdf
    zero = pg.fill((), p0.dtype, 0, p0.device)
    one = pg.fill((), p0.dtype, 1, p0.device)
    minus_one = pg.fill((), p0.dtype, -1, p0.device)

    def _arccos(x):
        # polynomial approximation for arccos(x) in [0, 1]
        x = where(x < -1, minus_one, x)
        x = where(x > 1, one, x)
        return (3.14159265 / 2) - (x + (x**3 / 6) + (3 * x**5 / 40))

    def _sin(x):
        # polynomial approximation for sin(x) in [0, 1]
        x = where(x < 0, zero, x)
        x = where(x > 1, one, x)
        return x - (x**3 / 6) + (x**5 / 120)

    angle = _arccos((p0 @ p1) / (pg.norm(p0) * pg.norm(p1)))
    return (_sin((1 - alpha) * angle) * p0 + _sin(alpha * angle) * p1) / _sin(angle)


def diag_vector(vector: Tensor) -> Tensor:
    """
    Returns a 2D tensor with the vector as the diagonal
    """
    I = eye(vector.shape[0], vector.shape[0], vector.dtype, vector.device)
    return I * vector  # will be broadcasted


def extract_diag(matrix: Tensor, ret_vec=False) -> Tensor:
    """
    Returns the diagonal of the matrix
    """
    assert matrix.ndim == 2, "extract_diag is only supported for 2D tensors"
    if not ret_vec:  # returns diag matrix
        return (
            eye(matrix.shape[0], matrix.shape[1], matrix.dtype, matrix.device) * matrix
        )
    else:  # returns vector
        diagmat = (
            eye(matrix.shape[0], matrix.shape[1], matrix.dtype, matrix.device) * matrix
        )
        return diagmat.sum(1)  # sum along the columns (0 + x = 0)


def extract_non_diag(matrix: Tensor) -> Tensor:
    """
    Returns the matrix without the diagonal
    """
    assert matrix.ndim == 2, "extract_non_diag is only supported for 2D tensors"
    return (
        fill(matrix.shape, matrix.dtype, 1, matrix.device)
        - eye(matrix.shape[0], matrix.shape[1], matrix.dtype, matrix.device)
    ) * matrix


def global_avg_pool2d(self) -> Tensor:
    """
    Returns the global average pooling of the tensor
    """
    assert self.ndim == 4, "expected a 4D tensor, got {}".format(self.ndim)
    return self.mean(dim=(2, 3))


def covariance_matrix(self) -> Tensor:
    """
    Returns the covariance matrix of the tensor
    Data should be in the form of a matrix or N-D tensor of the form (b1, ..., bN, samples, features)
    If rowvar is True, then each row is a variable, with observations in the columns.
    If rowvar is False, then each column is a variable, with observations in the rows.
    Any extra dimensions on the left are treated as batch dimensions.
    """
    demeaned = self - self.mean(dim=-2, keepdim=True)
    return (demeaned @ demeaned.transpose(-2, -1)) / (self.shape[-2] - 1)


cos = pg.cos

sin = pg.sin


def round(self):
    return (self + 0.5).astype(dt.int32).astype(self.dtype)


def clip(self, min, max):
    min, max = (
        (
            fill((), self.dtype, min, self.device)
            if isinstance(min, (int, float))
            else min
        ),
        (
            fill((), self.dtype, max, self.device)
            if isinstance(max, (int, float))
            else max
        ),
    )
    min, max = pg.broadcast_to(min, self.shape), pg.broadcast_to(max, self.shape)
    return pg.where(self < min, min, pg.where(self > max, max, self))


def argmax(self, dim: int = -1):
    max_ = pg.max_reduce(self, axes=dim, keepdims=True)
    reduced_shape = list(max_.shape)
    del reduced_shape[dim]
    assert len(reduced_shape) == 1
    # todo -- better
    ar = pg.reshape(arange(0, self.shape[1], 1, dt.float32, device.cpu(0)), (1, -1)).to(
        self.device
    )
    aranges = pg.broadcast_to(ar, (self.shape[0], self.shape[1]))
    return pg.sum((self == max_) * aranges, axes=dim)


def accuracy(self, target):
    return (_abs(self - target) < 1e-6).mean()


def repeat_new_dim(self, repeats: int, dim: int):
    """
    Repeats the tensor along a new dimension
    """
    dim = dim if dim >= 0 else self.ndim + dim + 1
    shape = list(self.shape)
    shape.insert(dim, repeats)
    shapewith1 = list(self.shape)
    shapewith1.insert(dim, 1)
    return pg.broadcast_to(self.reshape(shapewith1), shape)


def fill_like(self, value):
    return pg.fill(self.shape, self.dtype, value, self.device)


def ones_like(self):
    return fill_like(self, 1)


def ones(shape: _Shape, dtype=dt.float32, device="cpu"):
    return fill(shape, dtype, 1, DeviceModule.from_str(device))


def cumsum(self, dim: int = 0):
    N = self.shape[dim]
    t1 = self.transpose(dim, -1)
    other_shape = list(t1.shape)
    del other_shape[-1]
    t2 = (
        t1.to("cpu")
        .pad_constant((N - 1, 0), 0.0)
        .reshape((-1, 1, 1, 2 * N - 1))
        .to(self.device)
    )  # cuda is buggy on non contiguous pads
    t3 = t2.unfold((1, N))
    return (
        (t3).sum(dim=-1).reshape(other_shape + [N]).transpose(dim, -1)
    )  # shape (*old_shape)


def arange(start: int, end: int, step: int = 1, dtype=dt.float32, device="cpu"):
    if dtype == dt.int32:
        return (
            cumsum(ones(((end - start) // step,), dt.float32, device) * step, 0)
            + start
            - 1
        ).astype(dt.int32)
    return cumsum(ones(((end - start) // step,), dtype, device) * step, 0) + start - 1


def eye(n: int, m: int = None, dtype=dt.float32, device="cpu"):
    if m is None:
        m = n
    return (
        arange(0, n, 1, dtype, device).reshape((n, 1)) == arange(0, m, 1, dtype, device)
    ).astype(dtype)


def min_reduce(self, axes: int = None, keepdims: bool = False):
    return -pg.max_reduce(-self, axes, keepdims)


def digitize(self, nbins: int) -> Tensor:
    """
    Returns the indices of the bins to which each value in input belongs
    """
    assert self.ndim == 1, "digitize is only supported for 1D tensors"
    scale = (
        self.max_reduce() - self.min_reduce()
    ) / nbins  # in self, the size of each step is 'scale'
    return clip(
        round((self - self.min_reduce()) / scale).astype(dt.int32), 0, nbins - 1
    )


def histogram(digitized: Tensor, nbins) -> Tensor:
    """
    Returns the histogram of the tensor
    """
    assert digitized.ndim == 1, "histogram is only supported for 1D tensors"
    return pg.sum(
        arange(0, nbins, 1, digitized.dtype, digitized.device).reshape((1, -1))
        == digitized.reshape((-1, 1)),
        0,
    ).reshape((-1,))


def meshgrid(x, y):
    assert x.ndim == 1 and y.ndim == 1
    x = broadcast_to(x.reshape((-1, 1)), (x.shape[0], y.shape[0]))
    y = broadcast_to(y.reshape((1, -1)), (x.shape[0], y.shape[0]))
    return x, y


def linspace(start: float, end: float, steps: int, dtype=dt.float32, device="cpu"):
    return arange(0, steps, 1, dtype, device) * (end - start) / (steps - 1) + start


def floor(self):
    return round(self - 0.5)


def ceil(self):
    return round(self + 0.5)


def while_loop(cond, body, init):
    from pequegrad.transforms import vjp

    n_evals = 0

    class WhileLoop(extend.Primitive):
        @staticmethod
        def dispatch(inputs):
            val = inputs
            n_evals = 0
            while cond(*val).numpy() == 1:
                val = body(*val)
                n_evals += 1
            return tuple(map(lambda x: x.eval(), val))  # TODO -- multiple returns

        @staticmethod
        def backward(primals, tangents, outputs):
            # we need to do the backward pass of the body for each iteration
            tangents = tangents[0]  # only one output
            val = primals
            bodygrad = vjp(body, wrt=range(1, len(val)))
            # to evaluate the gradient, we'll go backwards
            # this is inefficient atm
            vals = []
            for i in range(n_evals):
                vals.append(val)
                val = body(*val)
            # now we have the values of the body at each iteration
            # we can start the backward pass
            for i in range(n_evals - 1, -1, -1):
                val = vals[i]
                val = bodygrad(val, tangents)
                tangents = val[0]
            return (
                fill_like(primals[0], 0),
                tangents,
            )  # iterator is not differentiable

    return WhileLoop.apply(*init)


def fori_loop(start, end, body, init):
    def cond(i, *args):
        return i < end

    def _body(i, *args):
        return i + 1, body(i, *args)

    return while_loop(cond, _body, (start,) + init)


def ifelse(cond, true_fn, false_fn, args):
    from pequegrad.transforms import vjp

    evaled = None

    class IfElse(extend.Primitive):
        @staticmethod
        def dispatch(inputs):
            assert len(inputs) == 1
            nonlocal evaled
            c = cond(*inputs).numpy()
            if c:
                evaled = "true"
                return true_fn(*inputs).eval()
            else:
                evaled = "false"
                return false_fn(*inputs).eval()

        @staticmethod
        def backward(primals, tangents, outputs):
            vjptrue = vjp(true_fn, wrt=[0])
            vjpfalse = vjp(false_fn, wrt=[0])
            if evaled == "true":
                return vjptrue(primals[0], tangents)
            elif evaled == "false":
                return vjpfalse(primals[0], tangents)
            else:
                raise ValueError("ifelse not evaluated")

    return IfElse.apply(*args)


def scan(scan_fn, init, seq):
    assert seq.ndim == 1, "scan is only supported for 1D tensors"

    # scan_fn is of the form (carry, curr) -> (new_carry, new_output)
    def _cond(i, *args):
        return i < seq.shape[0]

    def _body(i, *args):
        next_idx = i + 1
        assert len(args) == 2  # (carry, output)
        carry, acc = args
        new_carry, new_output = scan_fn(carry, seq[i])
        return next_idx, new_carry, acc.at[i].set(new_output)

    acc = fill_like(seq, 0).at[0].set(init)
    return while_loop(
        _cond,
        _body,
        (Tensor(0).astype("int32").to(seq.device).reshape((1,)),) + (init, acc),
    )[
        1:
    ]  # ret carry, output array
