from typing import List, Union, Type, Tuple, Optional
import numpy as np
from .context import pequegrad_context
from .backend import NumpyTensor, CudaTensor, CpuTensor, CUDA_AVAILABLE  # noqa: F401


_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]
_Backend = Union[NumpyTensor, CudaTensor, CpuTensor]


class Tensor:
    """
    Tensor implementation with autograd support
    """

    backend: _Backend
    _data: _Backend
    requires_grad: bool
    _grad: Type["Tensor"]
    _ctx: Optional["Function"]

    def contiguous(self):
        return self.data.contiguous()

    @property
    def dtype(self):
        return self.data.dtype

    def __init__(self, data: _ArrayLike, requires_grad=False, backend="np"):
        # Internally, we store the data as a numpy array\
        if isinstance(data, Tensor):
            if data.backend == "np" and backend == "np":
                data = data.data
            elif data.backend == "cuda" and backend == "cuda":
                data = data.data
            elif data.backend == "np" and backend == "cuda":
                data = data.data
            elif data.backend == "cuda" and backend == "np":
                data = data.numpy()
            elif data.backend == "cpu" and backend == "cpu":
                data = data.data
        elif isinstance(data, (CudaTensor, NumpyTensor, CpuTensor)):
            data = data.numpy() if backend == "np" else data
        elif isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, list):
            data = np.array(data)

        backend = backend if backend else "np"  # default to numpy backend
        if backend == "np":
            self._data: NumpyTensor = NumpyTensor(data)
        elif backend == "cuda":
            self._data: CudaTensor = CudaTensor(data)
        elif backend == "cpu":
            self._data: CpuTensor = CpuTensor(data)
        else:
            raise ValueError("backend must be 'np', 'cuda' or 'cpu'")

        # If the tensor was created under a no_grad context, it doesn't require gradients
        self.requires_grad: bool = requires_grad and pequegrad_context.grad_enabled

        self._grad: Type[Tensor] = None
        # The context is the function that created this tensor, along with its inputs. The function
        # is responsible for assigning itself to the _ctx attribute of the tensor
        self._ctx: Optional[Function] = None

    @property
    def data(self) -> _Backend:
        return self._data

    def assign(self, data: _Backend) -> None:
        """
        Assigns the tensor to the given data
        """
        assert isinstance(
            data, self._data.__class__
        ), "data must be of the same type as the current backend, expected {}, got {}".format(
            self._data.__class__, data.__class__
        )
        self._data = data

    def to(self, backend: str) -> "Tensor":
        """
        Moves the tensor to the given device, returns a copy
        """
        tensor = self.clone(backend=backend)
        return tensor

    def to_(self, backend: str) -> None:
        """
        Moves the tensor to the given device in place
        """
        # if the grad is not initialized, we don't need to move it
        if backend == "np":
            self._data = NumpyTensor(self.data.numpy())
            if self._grad is not None:
                self._grad.to_(backend)
        elif backend == "cuda":
            self._data = CudaTensor(self.data.numpy())
            if self._grad is not None:
                self._grad.to_(backend)

    @property
    def device(self) -> str:
        return self.backend

    __di = {
        CudaTensor: "cuda",
        NumpyTensor: "np",
        CpuTensor: "cpu",
    }

    @property
    def backend(self) -> str:
        return self.__di[self.data.__class__]

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data.numpy()}, fn={self._ctx.__class__.__name__ if self._ctx else None}, requires_grad={self.requires_grad}, backend={self.backend}), dtype={self.dtype}"

    def __getitem__(self, key):
        return Slice.apply(self, key=key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def backward(
        self, gradient: Type["Tensor"] = None, retain_ctx: bool = False
    ) -> None:
        """
        Backpropagate the gradient through the graph.
        If gradient is not provided, the gradient of the output node is assumed to be 1.0
        if the output node is a scalar. If the output node is not a scalar, gradient must be provided.

        Args:
            gradient: The gradient of the output node
            retain_ctx: Whether to retain the context after backpropagation. If it is set to false, the context (parent nodes)
                as well as the information about the function that created this tensor will be gone after backpropagation.
                This reduces memory usage, but makes it impossible to call backward again on this tensor.
        """

        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that doesn't require grad")

        # We start with the gradient of the output node, defaulting to 1.0 if it's a scalar
        # This is called implicit gradient creation
        self._grad = (
            gradient
            if gradient is not None
            else Tensor(1.0, backend=self.backend)
            if self.dim == 0
            else None
        )

        if not isinstance(self._grad, Tensor):
            raise TypeError(
                f"gradient must be a tensor, not {type(self._grad).__name__}"
            )

        # We assure that backward is only called once on _ctx per backward pass.
        visited = []
        nodes = []

        def _dfs(node):
            visited.append(node)
            for child in node._ctx.children if node._ctx else []:
                if child not in visited:
                    _dfs(child)
            nodes.append(node)

        _dfs(self)

        for node in reversed(nodes):
            if node._ctx is not None and node.requires_grad:
                grads = node._ctx.backward(node._grad.data)
                grads = (
                    [Tensor(g, backend=self.backend) for g in grads if g is not None]
                    if isinstance(grads, tuple)
                    else [Tensor(grads, backend=self.backend)]
                    if grads is not None
                    else []
                )
                for child, grad in zip(node._ctx.children, grads):
                    if grad is not None:
                        if child._grad is None:
                            child._grad = grad
                        else:
                            child._grad += grad
                assert (
                    node._grad.shape == node.shape
                ), f"gradient shape {node._grad.shape} does not match tensor shape {node.shape}, tensor: {node}"
                if not retain_ctx:
                    del node._ctx
                    del node._grad

    @property
    def grad(self):
        return self._grad

    def show_graph(self):
        from .graph import build_graph
        import matplotlib.pyplot as plt
        import networkx as nx

        G = build_graph(self)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2500)
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.show()

    def tolist(self):
        """
        Returns the tensor as a nested list. If the tensor is a scalar, returns the scalar
        """

        return self.data.numpy().tolist()

    def clone(self, requires_grad: bool = None, backend=None) -> "Tensor":
        """
        Returns an independent copy of the tensor.
        If requires_grad is not provided, the clone will have the same requires_grad as the original.

        Args:
            requires_grad: Whether the cloned tensor requires gradients

        Returns:
            An independent copy of the tensor
        """

        t = Tensor(
            self.numpy().copy(),
            requires_grad=(
                requires_grad if requires_grad is not None else self.requires_grad
            ),
            backend=backend if backend is not None else self.backend,
        )
        return t

    def reset_grad(self):
        """
        Resets the gradient of the tensor to 'None'
        """
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call reset_grad on a tensor that doesn't require grad"
            )
        self._grad = None

    ##### INITIALIZATION METHODS #####

    @classmethod
    def normal(cls, shape: _Shape, mean=0.0, std=1.0, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return cls(np.random.normal(mean, std, shape), requires_grad=requires_grad)

    @classmethod
    def uniform(
        cls, shape: _Shape, low=0.0, high=1.0, requires_grad=False, dtype="float32"
    ) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return cls(
            np.random.uniform(low, high, shape).astype(dtype),
            requires_grad=requires_grad,
        )

    @classmethod
    def randn(cls, shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape: _Shape, requires_grad=False, backend=None) -> "Tensor":
        """Returns a tensor of zeros with the given shape"""
        return cls.fill(shape, 0.0, requires_grad=requires_grad, backend=backend)

    @classmethod
    def ones(cls, shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        return cls.fill(shape, 1.0, requires_grad=requires_grad)

    @classmethod
    def fill(
        cls, shape: _Shape, value: float, requires_grad=False, backend=None
    ) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        st = NumpyTensor if backend == "np" or backend is None else CudaTensor
        return cls(st.fill(shape, value), requires_grad=requires_grad, backend=backend)

    def astype(self, dtype: str) -> "Tensor":
        """
        Returns a tensor with the given dtype
        """

        return Tensor(
            self.data.astype(dtype),
            requires_grad=self.requires_grad,
            backend=self.backend,
        )

    @classmethod
    def one_hot(
        cls,
        num_classes: int,
        indices: "Tensor",
        requires_grad=False,
        backend: str = "np",
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

        return cls(np_one_hot, requires_grad=requires_grad, backend=backend)

    def numpy(self) -> np.ndarray:
        return self.data.numpy()

    def reshape(self, shape: _Shape) -> "Tensor":
        """
        Returns a tensor with the given shape
        """
        return Reshape.apply(self, shape=shape)

    def transpose(self, dim0=0, dim1=1) -> "Tensor":
        """Transpose the tensor"""
        axis = list(range(self.dim))
        axis[dim0], axis[dim1] = axis[dim1], axis[dim0]

        return self.permute(*axis)

    def permute(self, *dims) -> "Tensor":
        """Permute the tensor"""
        return Permute.apply(self, dims=dims)

    def sum(self, dim: Optional[_Shape] = None, keepdim: bool = False) -> "Tensor":
        return Sum.apply(self, dim=dim, keepdim=keepdim)

    def exp(self) -> "Tensor":
        return Exp.apply(self)

    def mse_loss(a: "Tensor", b: "Tensor") -> "Tensor":
        return Pow.apply(a - b, 2).sum(None) / float(a.data.size)

    def pow(self, exponent: Union[float, int, "Tensor"]) -> "Tensor":
        """Returns the tensor raised to the given exponent"""
        return Pow.apply(self, exponent)

    def log(self) -> "Tensor":
        """Returns the natural logarithm of the tensor"""

        return Log.apply(self)

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Returns the maximum value of the tensor"""
        return Max.apply(self, dim=dim, keepdim=keepdim)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """Returns the mean value of the tensor"""
        return Mean.apply(self, dim=dim, keepdim=keepdim)

    def mul(self, other: "Tensor") -> "Tensor":
        """Hadamard product"""
        return Mul.apply(self, other)

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

    def pad_constant(self, pad: _Shape, constant=0):
        return PadConstant.apply(self, pad=pad, constant=constant)

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
        return -(target * self.log_softmax(dim=c_idx)).sum(c_idx).mean()

    def cross_entropy_loss_indices(self, target: "Tensor") -> "Tensor":
        """
        Returns the cross entropy loss of the tensor.
        Only works for inputs of shape (batch, C), and targets of shape (batch,)
        """

        assert self.dim == 2, "input must be a matrix, of shape (batch, C)"
        assert target.dim == 1, "target must be a vector, of shape (batch,)"
        assert (
            self.shape[0] == target.shape[0]
        ), "input and target must have the same batch size"

        one_hot_target = Tensor.one_hot(self.shape[1], target, backend=self.backend)

        return self.cross_entropy_loss_probs(one_hot_target)

    def unsqueeze(self, dim: int) -> "Tensor":
        """Returns a tensor with a dimension of size one inserted at the specified position"""
        return Unsqueeze.apply(self, dim=dim)

    def squeeze(self, dim: int) -> "Tensor":
        """Returns a tensor with the specified dimension removed"""
        return Squeeze.apply(self, dim=dim)

    @property
    def T(self) -> "Tensor":
        """Returns the transpose of the tensor"""
        return self.transpose(0, 1)

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
            (1, 1), after_conv_size
        )  # dilation and strides are implicitly 1

        if bias is not None:
            assert bias.shape == (
                out.shape[1],
            ), "bias shape must match output shape. Got {} but expected {}".format(
                bias.shape, (out.shape[1],)
            )
            out += bias.reshape((1, -1, 1, 1))  # so we add the bias to each channel
        return out

    def max_pool2d(
        self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = None
    ) -> "Tensor":
        """Returns the 2d maxpooling of the tensor with the given kernel size"""
        assert (
            self.dim == 4
        ), "max_pool2d is only supported for tensors with 4 dimensions"
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
        assert (
            self.dim == 4
        ), "avg_pool2d is only supported for tensors with 4 dimensions"
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
        ), "local_response_norm is only supported for tensors with 4 dimensions"

        # input of shape (batch, in_channels, height, width)
        # we need to normalize accross channels

        # first, pad the input so that we can calculate the normalization for the borders
        # (so the normalized output has the same shape as the input)
        pad = (size - 1) // 2
        padded = self.pad_constant((pad, pad, pad, pad), constant=0)

        # shape self: (batch, in_channels, height, width) -> (batch, size * size * in_channels, height, width)
        unfolded = padded.unfold((size, size), stride=1).reshape(
            (self.shape[0], -1, self.shape[2], self.shape[3])
        )

        # shape unfolded: (batch, size * size * in_channels, height, width) -> (batch, 1, height, width)
        norm_factor = (unfolded**2).mean(1, keepdim=True) * alpha + k

        return self / norm_factor**beta

    def unfold(
        self,
        kernel_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):
        return Unfold.apply(
            self, kernel_shape=kernel_shape, stride=stride, dilation=dilation
        )

    def fold(
        self,
        kernel_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
    ):
        return Fold.apply(
            self,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            stride=stride,
            dilation=dilation,
        )

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
            self.data.size
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

    # we pass training here to allow user to MANUALLY control dropout
    # but in modules.Dropout, we will use the context manager to control dropout
    def dropout(self, p=0.5, training=True):
        """Applies dropout to the input"""
        if training:
            mask = Tensor(
                np.random.binomial(1, 1 - p, size=self.shape), backend=self.backend
            ) / (1 - p)
            return self * mask
        else:
            return self

    @property
    def shape(self):
        """Returns the shape of the tensor"""
        return self.data.shape

    @property
    def dim(self):
        """Returns the number of dimensions of the tensor"""
        return len(self.shape)

    def __add__(self, other: "Tensor") -> "Tensor":
        return Add.apply(self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Hadamard product"""
        return self.mul(other)

    def __rmul__(self, other: "Tensor" or float) -> "Tensor":
        """Hadamard product"""
        return self.mul(other)

    def __neg__(self) -> "Tensor":
        return self.__mul__(-1.0)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self.__add__(other.__neg__())

    def __pow__(self, exponent: Union[float, int, "Tensor"]) -> "Tensor":
        return self.pow(exponent)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return Div.apply(self, other)

    def relu(self) -> "Tensor":
        """ReLU activation function"""
        return ReLU.apply(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        return MatMul.apply(self, other)

    def __len__(self):
        return len(self.data)


from .autodiff import (  # noqa: E402 avoid circular imports
    Add,
    MatMul,
    ReLU,
    Pow,
    Log,
    Exp,
    Max,
    Mean,
    Mul,
    Sum,
    Reshape,
    Unsqueeze,
    Squeeze,
    Unfold,
    Fold,
    Function,
    Permute,
    Div,
    Slice,
    PadConstant,
)
