from typing import List, Union, Type, Tuple, Optional
import numpy as np
from .context import pequegrad_context
from .storage import AbstractStorage, NumpyStorage, CudaStorage, CUDA_AVAILABLE  # noqa

_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]


class Tensor:
    """
    Tensor implementation with autograd support
    """

    storage: AbstractStorage

    def __init__(self, data: _ArrayLike, requires_grad=False, storage="np"):
        # Internally, we store the data as a numpy array\
        if isinstance(data, Tensor):
            data = data.numpy() if storage == "np" else data
        elif isinstance(data, AbstractStorage):
            data = data.numpy() if storage == "np" else data
        elif isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, list):
            data = np.array(data)

        storage = storage if storage else "np"  # default to numpy storage
        if storage == "np":
            self.data: NumpyStorage = NumpyStorage(data)
        elif storage == "cuda":
            self.data: CudaStorage = CudaStorage(data)
        else:
            raise ValueError("storage must be 'np' or 'cuda'")

        # If the tensor was created under a no_grad context, it doesn't require gradients
        self.requires_grad: bool = requires_grad and pequegrad_context.grad_enabled

        # Gradient is initialized as 0.0 if requires_grad is True, None otherwise
        self._grad: Type[Tensor] = (
            Tensor.zeros(self.shape, requires_grad=False, storage=storage)
            if self.requires_grad
            else None
        )

        # The context is the function that created this tensor, along with its inputs. The function
        # is responsible for assigning itself to the _ctx attribute of the tensor
        self._ctx: Optional[Function] = None

    def to(self, storage_type: str) -> "Tensor":
        """
        Moves the tensor to the given device, returns a copy
        """
        tensor = self.clone(storage=storage_type)
        return tensor

    def to_(self, storage_type: str) -> None:
        """
        Moves the tensor to the given device in place
        """
        if storage_type == "np":
            self.data = NumpyStorage(self.data.numpy())
            if self._grad:
                self._grad.to_(storage_type)
        elif storage_type == "cuda":
            self.data = CudaStorage(self.data.numpy())
            if self._grad:
                self._grad.to_(storage_type)

    @property
    def device(self) -> str:
        return "cuda" if isinstance(self.data, CudaStorage) else "np"

    @property
    def storage_type(self) -> str:
        return "cuda" if isinstance(self.data, CudaStorage) else "np"

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data.numpy()}, fn={self._ctx.__class__.__name__ if self._ctx else None}, requires_grad={self.requires_grad}, storage={self.storage_type})"

    def __getitem__(self, key):
        if self.data.ndim == 0:
            raise IndexError("0-d tensors cannot be indexed")

        return self.data[key]

    def backward(self, gradient: Type["Tensor"] = None, retain_ctx: bool = False):
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
            else Tensor(1.0, storage=self.storage_type)
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
                node._ctx.backward()
                assert (
                    node._grad.shape == node.shape
                ), f"gradient shape {node._grad.shape} does not match tensor shape {node.shape}"

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

    def clone(self, requires_grad: bool = None, storage=None) -> "Tensor":
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
            storage=storage if storage is not None else self.storage_type,
        )
        return t

    def zero_grad(self):
        """
        Zeroes the gradient of the tensor
        """
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call zero_grad on a tensor that doesn't require grad"
            )

        self._grad = Tensor.zeros(
            self.shape, requires_grad=False, storage=self.storage_type
        )

    ##### INITIALIZATION METHODS #####

    @classmethod
    def normal(cls, shape: _Shape, mean=0.0, std=1.0, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return cls(np.random.normal(mean, std, shape), requires_grad=requires_grad)

    @classmethod
    def uniform(cls, shape: _Shape, low=0.0, high=1.0, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return cls(np.random.uniform(low, high, shape), requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape: _Shape, requires_grad=False, storage=None) -> "Tensor":
        """Returns a tensor of zeros with the given shape"""
        return cls.fill(shape, 0.0, requires_grad=requires_grad, storage=storage)

    @classmethod
    def ones(cls, shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        return cls.fill(shape, 1.0, requires_grad=requires_grad)

    @classmethod
    def fill(
        cls, shape: _Shape, value: float, requires_grad=False, storage=None
    ) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        st = CudaStorage if storage == "cuda" else NumpyStorage
        return cls(st.fill(shape, value), requires_grad=requires_grad, storage=storage)

    @classmethod
    def one_hot(
        cls,
        num_classes: int,
        indices: "Tensor",
        requires_grad=False,
        storage_type: str = "np",
    ) -> "Tensor":
        assert indices.dim == 1, "indices must be a vector"
        assert np.all(indices.numpy() >= 0), "indices must be positive integers (>= 0)"
        assert np.all(
            indices.numpy() < num_classes
        ), "indices must be smaller than num_classes"

        np_one_hot = np.zeros((indices.data.size, num_classes))

        np_one_hot[np.arange(indices.data.size), indices.numpy().astype(int)] = 1.0

        return cls(np_one_hot, requires_grad=requires_grad, storage=storage_type)

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

        one_hot_target = Tensor.one_hot(
            self.shape[1], target, storage_type=self.storage_type
        )

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

    def conv2d(self, filter: "Tensor", bias: "Tensor" = None) -> "Tensor":
        """Returns the 2d convolution of the tensor with the given filter"""
        inp_unf = self.unfold(filter.shape[-2:])
        out_unf = (
            inp_unf.transpose(1, 2) @ filter.reshape((filter.shape[0], -1)).T
        ).transpose(1, 2)
        after_conv_size = (
            self.shape[-2] - filter.shape[-2] + 1,
            self.shape[-1] - filter.shape[-1] + 1,
        )
        out = out_unf.fold((1, 1), after_conv_size)

        if bias is not None:
            assert bias.shape == (
                out.shape[1],
            ), "bias shape must match output shape. Got {} but expected {}".format(
                bias.shape, (out.shape[1],)
            )
            out += bias.reshape((1, -1, 1, 1))  # so we add the bias to each channel

        return out

    def max_pool2d(self, kernel_size: Tuple[int, int]) -> "Tensor":
        """Returns the 2d maxpooling of the tensor with the given kernel size"""
        assert (
            self.dim == 4
        ), "max_pool2d is only supported for tensors with 4 dimensions"
        stride = kernel_size
        assert stride[0] == stride[1], "kernel size must be square"

        new_shape = (
            self.shape[0],
            self.shape[1],
            (self.shape[2] - kernel_size[0]) // stride[0] + 1,
            (self.shape[3] - kernel_size[1]) // stride[1] + 1,
        )
        unfolded = self.unfold(kernel_size, stride=stride[0])
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

    def unfold(self, kernel_shape: Tuple[int, ...], stride: int = 1):
        return Unfold.apply(self, kernel_shape=kernel_shape, stride=stride)

    def fold(
        self,
        kernel_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        stride: int = 1,
    ):
        return Fold.apply(
            self, kernel_shape=kernel_shape, output_shape=output_shape, stride=stride
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
        assert (
            self.shape[-ns_l:] == normalized_shape
        ), "normalized_shape should be the last dimensions of input tensor"

        last_d_dims = tuple(range(self.dim - ns_l, self.dim))

        mean = self.mean(dim=last_d_dims, keepdim=True)
        variance = self.var(
            dim=last_d_dims, keepdim=True, correction=0
        )  # unbiased variance is used
        # for numerical stability, we add eps before sqrt
        std = (variance + eps).sqrt()

        return (self - mean) / std

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
        return self.__mul__(other**-1.0)

    def relu(self) -> "Tensor":
        """ReLU activation function"""
        return ReLU.apply(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        return MatMul.apply(self, other)

    def __len__(self):
        return len(self.data)


from .function import (  # noqa: E402 avoid circular imports
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
)
