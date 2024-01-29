from typing import Optional, Set, Tuple, Union
from .tensor import Tensor
from .util import im2col, col2im
from .context import pequegrad_context
import numpy as np


_Shape = Union[int, Tuple[int, ...]]


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
        should_store_grad = f.requires_grad and pequegrad_context.grad_enabled
        if should_store_grad:
            f.ret._ctx = f

        return f.ret


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
            grad_output = self.ret.grad.data
            ret_data = self.ret.data
            # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
            # so that broadcasting works correctly during the backward pass.
            if not self.keepdim and self.dim is not None:
                grad_output = np.expand_dims(grad_output, axis=self.dim)
                ret_data = np.expand_dims(ret_data, axis=self.dim)

            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = np.broadcast_to(grad_output, self.a.shape)
            ret_broadcasted = np.broadcast_to(ret_data, self.a.shape)

            self.a._grad += Tensor(
                np.where(self.a.data == ret_broadcasted, grad_broadcasted, 0)
            )


class Unsqueeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            np.expand_dims(self.a.data, axis=self.dim), requires_grad=self.requires_grad
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(np.squeeze(self.ret.grad.data, axis=self.dim))


class Squeeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            np.squeeze(self.a.data, axis=self.dim), requires_grad=self.requires_grad
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(np.expand_dims(self.ret.grad.data, axis=self.dim))


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


class Permute(Function):
    def __init__(self, a: Tensor, dims: Tuple[int, ...]):
        super().__init__(a)
        self.a = a
        self.dims = dims

    def forward(self):
        self.ret = Tensor(
            np.transpose(self.a.data, self.dims),
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self):
        bw_dims = np.argsort(
            self.dims
        )  # computes the indices that would sort the dims back
        if self.a.requires_grad:
            self.a._grad += Tensor(
                np.transpose(self.ret.grad.data, bw_dims),
            )


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
            self.x.data + self.y.data,
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
        # If we are multiplying 2 matrices, we need to check out cases where one of them is batched
        # ,of shape (...extra_dims, m, n) and the other is not (of shape (n, k)).
        # In that case, we need to sum over the batch dimensions to get the gradient of the non-batched matrix
        # so the gradient has the same shape as the non-batched matrix
        # TODO -- check cases with vectors, and higher dimensional tensors (rn it works when shape of the batched matrix is (batch, m, n) and the other is (n, k))
        if self.x.requires_grad:
            if self.x.dim == 1 and self.y.dim == 1:
                # Just multiply the gradients if both are vectors, since grad is a scalar
                self.x._grad += Tensor(np.multiply(grad_output, self.y.data))
            elif self.x.dim == 1:
                # Vector x Matrix
                self.x._grad += Tensor(grad_output @ self.y.data.T)
            elif self.y.dim == 1:
                # Matrix x Vector
                diff = self.y.dim - self.x.dim
                axis_to_sum_over = tuple(range(diff))
                self.x._grad += Tensor(
                    np.outer(grad_output, self.y.data).sum(axis=axis_to_sum_over)
                )
            else:
                # Matrix x Matrix
                diff = self.y.dim - self.x.dim
                axis_to_sum_over = tuple(range(diff))
                self.x._grad += Tensor(
                    (grad_output @ self.y.data.swapaxes(-1, -2)).sum(
                        axis=axis_to_sum_over
                    )
                )

        if self.y.requires_grad:
            if self.x.dim == 1 and self.y.dim == 1:
                self.y._grad += Tensor(np.multiply(self.x.data, grad_output))
            elif self.x.dim == 1:
                self.y._grad += Tensor(np.outer(self.x.data, grad_output))
            elif self.y.dim == 1:
                # Matrix x Vector
                self.y._grad += Tensor(self.x.data.T @ grad_output)
            else:
                # Matrix x Matrix
                diff = self.x.dim - self.y.dim
                axis_to_sum_over = tuple(range(diff))
                self.y._grad += Tensor(
                    (self.x.data.swapaxes(-1, -2) @ grad_output).sum(
                        axis=axis_to_sum_over
                    )
                )

        if self.x.requires_grad:
            assert self.x._grad.shape == self.x.shape, (
                f"grad shape {self.x._grad.shape} does not match tensor shape {self.x.shape}"
                + f"\ngrad_output shape: {grad_output.shape}"
            )
        if self.y.requires_grad:
            assert self.y._grad.shape == self.y.shape, (
                f"grad shape {self.y._grad.shape} does not match tensor shape {self.y.shape}"
                + f"\ngrad_output shape: {grad_output.shape}"
            )


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


class Unfold(Function):
    def __init__(self, input: Tensor, kernel_shape: Tuple[int, ...], stride: int = 1):
        super().__init__(input)
        self.input = input
        self.kernel_shape = kernel_shape
        self.stride = stride

    def forward(self) -> Tensor:
        unfolded = im2col(self.input.data, self.kernel_shape, stride=self.stride)
        self.ret = Tensor(
            unfolded,
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            folded_grad = col2im(
                self.ret.grad.data,
                self.kernel_shape,
                self.input.shape[-2:],
                stride=self.stride,
            )
            self.input._grad += Tensor(folded_grad)


class Fold(Function):
    def __init__(
        self,
        input: Tensor,
        kernel_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        stride: int = 1,
    ):
        super().__init__(input)
        self.input = input
        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.stride = stride

    def forward(self) -> Tensor:
        folded = col2im(
            self.input.data,
            self.kernel_shape,
            self.output_shape,
            stride=self.stride,
        )
        self.ret = Tensor(
            folded,
            requires_grad=self.requires_grad,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            unfolded = im2col(self.ret.grad.data, self.kernel_shape, stride=self.stride)
            self.input._grad += Tensor(unfolded)
