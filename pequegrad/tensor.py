from typing import List, Union, Type, Set, Tuple, Optional
import numpy as np

_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]


class Tensor:
    """
    Tensor implementation with autograd support
    """

    def __init__(self, data: _ArrayLike, requires_grad=False):
        # Internally, we store the data as a numpy array
        data = (
            np.array(data)
            if not isinstance(data, np.ndarray) and not isinstance(data, Tensor)
            else data.data
            if isinstance(data, Tensor)
            else data
        )

        self.data: np.ndarray = data

        # Gradient is initialized as 0.0 if requires_grad is True, None otherwise
        self._grad: Type[Tensor] = (
            Tensor.zeros(self.shape, requires_grad=False) if requires_grad else None
        )

        # The context is the function that created this tensor, along with its inputs
        self._ctx: Optional[Function] = None

        self.requires_grad: bool = requires_grad

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data}, fn={self._ctx.__class__.__name__ if self._ctx else None}, requires_grad={self.requires_grad})"

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
            gradient if gradient is not None else Tensor(1.0) if self.dim == 0 else None
        )

        if not isinstance(self._grad, Tensor):
            raise TypeError(
                f"gradient must be a tensor, not {type(self._grad).__name__}"
            )

        visited = []

        def _dfs(node):
            if node not in visited:
                visited.append(node)
                for child in node._ctx.children if node._ctx else []:
                    _dfs(child)

        _dfs(self)

        for node in visited:
            if node._ctx is not None:
                node._ctx.backward()
                if not retain_ctx:
                    node._ctx = None

    @property
    def grad(self):
        return self._grad

    def tolist(self):
        """
        Returns the tensor as a nested list. If the tensor is a scalar, returns the scalar
        """

        return self.data

    # google style comments
    def clone(self, requires_grad: bool = None):
        """
        Returns an independent copy of the tensor.
        If requires_grad is not provided, the clone will have the same requires_grad as the original.

        Args:
            requires_grad: Whether the cloned tensor requires gradients

        Returns:
            An independent copy of the tensor
        """

        return Tensor(
            self.data.copy(),
            requires_grad=requires_grad
            if requires_grad is not None
            else self.requires_grad,
        )

    def zero_grad(self):
        """
        Zeroes the gradient of the tensor
        """
        if not self.requires_grad:
            raise RuntimeError(
                "cannot call zero_grad on a tensor that doesn't require grad"
            )

        self._grad = Tensor.zeros(self.shape)

    ##### INITIALIZATION METHODS #####
    @staticmethod
    def randn(shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of random numbers with the given shape"""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def zeros(shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of zeros with the given shape"""
        return Tensor.fill(shape, 0.0, requires_grad=requires_grad)

    @staticmethod
    def ones(shape: _Shape, requires_grad=False) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        return Tensor.fill(shape, 1.0, requires_grad=requires_grad)

    @staticmethod
    def fill(shape: _Shape, value: float, requires_grad=False) -> "Tensor":
        """Returns a tensor of ones with the given shape"""
        return Tensor(np.full(shape, value), requires_grad=requires_grad)

    def reshape(self, shape: _Shape) -> "Tensor":
        """
        Returns a tensor with the given shape
        """
        return Reshape.apply(self, shape=shape)

    def transpose(self, dim0=0, dim1=1) -> "Tensor":
        """Transpose the tensor"""
        return Transpose.apply(self, dim0=dim0, dim1=dim1)

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

    def _softmax_helper(self, dim) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """Returns the softmax of the tensor"""
        normalized = self - self.max(dim=dim, keepdim=True)
        exponentiated = normalized.exp()
        summed = exponentiated.sum(dim=dim, keepdim=True)
        return (
            normalized,
            exponentiated,
            summed,
        )

    def softmax(self, dim=-1) -> "Tensor":
        """Returns the softmax of the tensor"""
        normalized, exponentiated, summed = self._softmax_helper(dim)
        return exponentiated / summed

    def log_softmax(self, dim=-1) -> "Tensor":
        """Returns the log softmax of the tensor"""
        normalized, exponentiated, summed = self._softmax_helper(dim)
        return normalized - summed.log()

    def logsumexp(self, dim=-1) -> "Tensor":
        """Returns the log sum exp of the tensor"""
        return self._softmax_helper(dim)[2].log()

    def cross_entropy_loss(self, target: "Tensor") -> "Tensor":
        """Returns the cross entropy loss of the tensor"""
        # At the moment, we assume minibatch affects shape like (batch_size, num_classes)
        scaled_logits = self - self.max(dim=-1, keepdim=True)
        normalized_logits = scaled_logits - scaled_logits.logsumexp()
        return -(target * normalized_logits).sum(dim=-1).mean()

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
        # Multiplying two tensors mean multiplying their elements, so their shapes must match
        return self.__mul__(other)

    def __neg__(self) -> "Tensor":
        return self.__mul__(-1.0)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self.__add__(other.__neg__())

    def __pow__(self, exponent: Union[float, int, "Tensor"]) -> "Tensor":
        return self.pow(exponent)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return self.__mul__(other**-1)

    def relu(self) -> "Tensor":
        """ReLU activation function"""
        return ReLU.apply(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        return MatMul.apply(self, other)


class Function:
    ret: Tensor
    children: Set[Tensor]
    requires_grad: bool

    def __init__(self, *tensors: Tensor):
        assert all(
            isinstance(t, Tensor) for t in tensors
        ), "all inputs must be tensors, got: {}".format(
            list(filter(lambda t: not isinstance(t, Tensor), tensors))
        )
        self.requires_grad = any(t.requires_grad for t in tensors)
        self.children = set(t for t in tensors if t.requires_grad)

    def forward(self):
        raise NotImplementedError

    def backward(self) -> Tensor:
        raise NotImplementedError

    @classmethod
    def apply(
        cls,
        *tensors: Tensor,
        **kwargs,
    ) -> Tensor:
        tensors = [Tensor(t) if not isinstance(t, Tensor) else t for t in tensors]

        f = cls(*tensors, **kwargs)
        f.forward()
        f.ret._ctx = f
        return f.ret


# TODO -- implement with dim argument
class Max(Function):
    def __init__(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            np.max(self.a.data, axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += (
                Tensor(
                    np.where(self.a.data == self.ret.data, 1, 0),
                )
                * self.ret.grad
            )


class Mean(Function):
    def __init__(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            np.mean(self.a.data, axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            grad_output = self.ret.grad.data
            # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
            # so that broadcasting works correctly during the backward pass.
            if not self.keepdim and self.dim is not None:
                grad_output = np.expand_dims(grad_output, axis=self.dim)
            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = np.broadcast_to(grad_output, self.a.shape)

            # now we need to divide the gradient by the number of elements WE SUMMED OVER(not all elements)
            # TODO -- optimize this
            total_els = 1
            if self.dim is None:
                total_els = self.a.data.size
            elif isinstance(self.dim, int):
                total_els = self.a.shape[self.dim]
            else:
                total_els = 1
                for d in self.dim:
                    total_els *= self.a.shape[d]
            self.a._grad += Tensor(grad_broadcasted) / total_els


class Sum(Function):
    def __init__(self, a: Tensor, dim: Optional[_Shape] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            np.sum(self.a.data, axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(
        self,
    ):
        if self.a.requires_grad:
            grad_output = self.ret.grad.data
            # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
            # so that broadcasting works correctly during the backward pass.
            if not self.keepdim and self.dim is not None:
                grad_output = np.expand_dims(grad_output, axis=self.dim)
            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = np.broadcast_to(grad_output, self.a.shape)
            self.a._grad += Tensor(grad_broadcasted)


class Pow(Function):
    def __init__(self, base: Tensor, exponent: Union[float, int, Tensor]):
        self.base = base
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        self.exponent = exponent

        super().__init__(base, exponent)

    def forward(self):
        self.ret = Tensor(
            np.power(self.base.data, self.exponent.data),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        if self.base.requires_grad:
            self.base._grad += Tensor(
                self.ret.grad.data
                * self.exponent.data
                * np.power(self.base.data, self.exponent.data - 1)
            )
        if self.exponent.requires_grad:
            self.exponent._grad += Tensor(
                self.ret.grad.data * self.ret.data * np.log(self.base.data)
            )


class Log(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a

    def forward(self):
        self.ret = Tensor(
            np.log(self.a.data),
            requires_grad=self.requires_grad,
        )

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad / self.a.data


class Exp(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a

    def forward(self):
        self.ret = Tensor(
            np.exp(self.a.data),
            requires_grad=self.requires_grad,
        )

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad * Tensor(np.exp(self.a.data))


class Transpose(Function):
    def __init__(self, a: Tensor, dim0: int, dim1: int):
        super().__init__(a)
        self.a = a
        self.dim0 = dim0
        self.dim1 = dim1

        # We swap the dimensions in the axes list
        axes = list(range(self.a.dim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        self.axes = axes

    def forward(self):
        self.ret = Tensor(
            np.transpose(self.a.data, self.axes),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad.transpose(self.dim0, self.dim1)


class ReLU(Function):
    """Implements the ReLU activation function: ReLU(x) = max(0, x)"""

    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            np.maximum(self.a.data, 0),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        # grad = 1 if a > 0 else 0
        if self.a.requires_grad:
            self.a._grad += (
                Tensor(
                    np.where(self.a.data > 0, 1, 0),
                )
                * self.ret.grad
            )


class Add(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            np.add(self.x.data, self.y.data),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        # If, for example, x was shape (200) and y was shape (32, 200), in the forward pass we "broadcasted" x to shape (32, 200) by repeating it 32 times along the first axis.
        # Since the gradient must be the same shape as the input, we must sum the gradient along the first axis to get the gradient of x in the backward pass if this was the case.
        # Same goes for y if x was shape (32, 200) and y was shape (200)
        if self.x.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sx, sy) in enumerate(zip(self.x.shape, grad_output.shape))
                if sx != sy
            ]

            grad = grad_output.sum(axis=tuple(axes_to_sum), keepdims=True)

            self.x._grad += Tensor(grad).reshape(self.x.shape)

        if self.y.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sy, sx) in enumerate(zip(self.y.shape, grad_output.shape))
                if sy != sx
            ]

            grad = grad_output.sum(axis=tuple(axes_to_sum), keepdims=True)

            self.y._grad += Tensor(grad).reshape(self.y.shape)


class Mul(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            np.multiply(self.x.data, self.y.data),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        # If, for example, x was shape (200) and y was shape (32, 200), in the forward pass we "broadcasted" x to shape (32, 200) by repeating it 32 times along the first axis.
        # Since the gradient must be the same shape as the input, we must sum the gradient along the first axis to get the gradient of x in the backward pass if this was the case.
        # Same goes for y if x was shape (32, 200) and y was shape (200)
        if self.x.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sx, sy) in enumerate(zip(self.x.shape, grad_output.shape))
                if sx != sy
            ]

            grad = grad_output * self.y.data
            grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
            self.x._grad += Tensor(grad).reshape(self.x.shape)

        if self.y.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sy, sx) in enumerate(zip(self.y.shape, grad_output.shape))
                if sy != sx
            ]
            grad = grad_output * self.x.data
            grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
            self.y._grad += Tensor(grad).reshape(self.y.shape)


class MatMul(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.ret: Tensor = None
        self.grad: Tensor = None

    def forward(self):
        self.ret = Tensor(
            np.matmul(self.x.data, self.y.data),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        if self.x.requires_grad:
            if self.x.dim == 1 and self.y.dim == 1:
                # Just multiply the gradients if both are vectors, since grad is a scalar
                self.x._grad = Tensor(np.multiply(grad_output, self.y.data))
            elif self.x.dim == 1:
                # Vector x Matrix
                self.x._grad = Tensor(grad_output @ self.y.data.T)
            elif self.y.dim == 1:
                # Matrix x Vector
                self.x._grad = Tensor(np.outer(grad_output, self.y.data))
            else:
                # Matrix x Matrix
                self.x._grad = Tensor(grad_output @ self.y.data.T)

        if self.y.requires_grad:
            if self.x.dim == 1 and self.y.dim == 1:
                self.y._grad = Tensor(np.multiply(self.x.data, grad_output))
            elif self.x.dim == 1:
                self.y._grad = Tensor(np.outer(self.x.data, grad_output))
            elif self.y.dim == 1:
                # Matrix x Vector
                self.y._grad = Tensor(self.x.data.T @ grad_output)
            else:
                # Matrix x Matrix
                self.y._grad = Tensor(self.x.data.T @ grad_output)

        if self.x.requires_grad:
            assert self.x._grad.shape == self.x.shape
        if self.y.requires_grad:
            assert self.y._grad.shape == self.y.shape


class Reshape(Function):
    def __init__(self, input: Tensor, shape: Tuple[int, ...]):
        super().__init__(input)
        self.input = input
        self.output_shape = shape
        self.input_shape = input.shape

    def forward(self) -> Tensor:
        self.ret = Tensor(
            np.reshape(
                self.input.data,
                self.output_shape,
            ),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            self.input._grad += Tensor(self.ret.grad.data).reshape(self.input_shape)
