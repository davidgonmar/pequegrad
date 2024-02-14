from typing import Optional, Set, Tuple, Union
from .tensor import Tensor
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
        # first, find first tensor that is a tensor
        device = "np"
        for t in tensors:
            if isinstance(t, Tensor):
                device = t.device
                break
        tensors = [
            Tensor(t, storage=device) if not isinstance(t, Tensor) else t
            for t in tensors
        ]
        # all devices should be the same
        assert all(
            t.device == device for t in tensors
        ), "all tensors must be on the same device, got: {}".format(
            [t.device for t in tensors]
        )

        cls.storage = device

        f = cls(*tensors, **kwargs)
        f.forward()
        should_store_grad = f.requires_grad and pequegrad_context.grad_enabled
        if should_store_grad:
            f.ret._ctx = f

        assert (
            f.ret.device == device
        ), f"function output device {f.ret.device} does not match input device {device}"

        return f.ret


class Max(Function):
    def __init__(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            self.a.data.max(axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            grad_output = self.ret.grad.data
            ret_data = self.ret.data
            # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
            # so that broadcasting works correctly during the backward pass.
            if not self.keepdim and self.dim is not None:
                grad_output = grad_output.expand_dims(axis=self.dim)
                ret_data = ret_data.expand_dims(axis=self.dim)

            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = grad_output.broadcast_to(self.a.shape)
            ret_broadcasted = ret_data.broadcast_to(self.a.shape)

            self.a._grad += Tensor(
                grad_broadcasted.where(self.a.data == ret_broadcasted, 0),
                storage=self.storage,
            )


class Unsqueeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            self.a.data.expand_dims(axis=self.dim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.squeeze(axis=self.dim), storage=self.storage
            )


class Squeeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            self.a.data.squeeze(axis=self.dim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.expand_dims(axis=self.dim), storage=self.storage
            )


class Mean(Function):
    def __init__(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            self.a.data.mean(axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            grad_output = self.ret.grad.data
            # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
            # so that broadcasting works correctly during the backward pass.
            if not self.keepdim and self.dim is not None:
                grad_output = grad_output.expand_dims(axis=self.dim)
            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = grad_output.broadcast_to(self.a.shape)

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
            self.a._grad += Tensor(grad_broadcasted, storage=self.storage) / total_els


class Sum(Function):
    def __init__(self, a: Tensor, dim: Optional[_Shape] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim

    def forward(self):
        self.ret = Tensor(
            self.a.data.sum(axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
            storage=self.storage,
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
                grad_output = grad_output.expand_dims(axis=self.dim)
            # Now we can broadcast the gradient to the shape of the input tensor
            grad_broadcasted = grad_output.broadcast_to(self.a.shape)

            self.a._grad += Tensor(grad_broadcasted, storage=self.storage)


class Pow(Function):
    def __init__(self, base: Tensor, exponent: Union[float, int, Tensor]):
        self.base = base
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent, requires_grad=False, storage=self.base.device)
        self.exponent = exponent

        super().__init__(base, exponent)

    def forward(self):
        self.ret = Tensor(
            self.base.data.power(self.exponent.data),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.base.requires_grad:
            self.base._grad += Tensor(
                self.ret.grad.data
                * self.exponent.data
                * self.base.data.power(self.exponent.data - 1),
                storage=self.storage,
            )
        if self.exponent.requires_grad:
            self.exponent._grad += Tensor(
                self.ret.grad.data * self.ret.data * self.base.data.log(),
                storage=self.storage,
            )


class Log(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a

    def forward(self):
        self.ret = Tensor(
            self.a.data.log(),
            requires_grad=self.requires_grad,
            storage=self.storage,
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
            self.a.data.exp(),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad * Tensor(self.ret.data, storage=self.storage)


class Permute(Function):
    def __init__(self, a: Tensor, dims: Tuple[int, ...]):
        super().__init__(a)
        self.a = a
        self.dims = dims

    def forward(self):
        self.ret = Tensor(
            self.a.data.permute(*self.dims),
            requires_grad=self.requires_grad,
            storage=self.a.device,
        )

        return self.ret

    def backward(self):
        bw_dims = np.argsort(
            self.dims
        )  # computes the indices that would sort the dims back
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.permute(*bw_dims), storage=self.storage
            )


class ReLU(Function):
    """Implements the ReLU activation function: ReLU(x) = max(0, x)"""

    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            self.a.data.el_wise_max(0),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        # grad = 1 if a > 0 else 0
        if self.a.requires_grad:
            concrete_class = self.a.data.__class__
            self.a._grad += (
                Tensor(
                    # steal the concrete class from tensor storage
                    concrete_class.where_static(self.a.data > 0, 1, 0),
                    requires_grad=False,
                    storage=self.storage,
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
            storage=self.x.device,
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

            self.x._grad += Tensor(grad, storage=self.storage).reshape(self.x.shape)

        if self.y.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sy, sx) in enumerate(zip(self.y.shape, grad_output.shape))
                if sy != sx
            ]

            grad = grad_output.sum(axis=tuple(axes_to_sum), keepdims=True)

            self.y._grad += Tensor(grad, storage=self.storage).reshape(self.y.shape)


class Mul(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            self.x.data * self.y.data,
            requires_grad=self.requires_grad,
            storage=self.x.device,
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
            self.x._grad += Tensor(grad, storage=self.storage).reshape(self.x.shape)

        if self.y.requires_grad:
            # Sum the gradient over axes that were broadcasted during the forward pass
            axes_to_sum = [
                i
                for i, (sy, sx) in enumerate(zip(self.y.shape, grad_output.shape))
                if sy != sx
            ]
            grad = grad_output * self.x.data
            grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
            self.y._grad += Tensor(grad, storage=self.storage).reshape(self.y.shape)


class MatMul(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.ret: Tensor = None
        self.grad: Tensor = None

    def forward(self):
        self.ret = Tensor(
            self.x.data @ self.y.data,
            requires_grad=self.requires_grad,
            storage=self.x.device,
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
                self.x._grad += Tensor(grad_output * self.y.data, storage=self.storage)
            elif self.x.dim == 1:
                # Vector x Matrix
                self.x._grad += Tensor(
                    grad_output @ self.y.data.T, storage=self.storage
                )
            elif self.y.dim == 1:
                # Matrix x Vector
                diff = self.y.dim - self.x.dim
                axis_to_sum_over = tuple(range(diff))
                self.x._grad += Tensor(
                    grad_output.outer_product(self.y.data).sum(axis=axis_to_sum_over),
                    storage=self.storage,
                )
            else:
                # Matrix x Matrix
                diff = self.y.dim - self.x.dim
                axis_to_sum_over = tuple(range(diff))
                self.x._grad += Tensor(
                    (grad_output @ self.y.data.swapaxes(-1, -2)).sum(
                        axis=axis_to_sum_over
                    ),
                    storage=self.storage,
                )

        if self.y.requires_grad:
            if self.x.dim == 1 and self.y.dim == 1:
                self.y._grad += Tensor(self.x.data * grad_output, storage=self.storage)
            elif self.x.dim == 1:
                self.y._grad += Tensor(
                    self.x.data.outer_product(grad_output), storage=self.storage
                )
            elif self.y.dim == 1:
                # Matrix x Vector
                self.y._grad += Tensor(
                    self.x.data.T @ grad_output, storage=self.storage
                )
            else:
                # Matrix x Matrix
                diff = self.x.dim - self.y.dim
                axis_to_sum_over = tuple(range(diff))
                self.y._grad += Tensor(
                    (self.x.data.swapaxes(-1, -2) @ grad_output).sum(
                        axis=axis_to_sum_over
                    ),
                    storage=self.storage,
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
            self.input.data.reshape(self.output_shape),
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            self.input._grad += Tensor(
                self.ret.grad.data, storage=self.storage
            ).reshape(self.input_shape)


class Unfold(Function):
    def __init__(self, input: Tensor, kernel_shape: Tuple[int, ...], stride: int = 1):
        super().__init__(input)
        self.input = input
        self.kernel_shape = kernel_shape
        self.stride = stride

    def forward(self) -> Tensor:
        unfolded =  self.input.to("np").data.im2col(
           self.kernel_shape, stride=self.stride
        )
        self.ret = Tensor(
            unfolded,
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            folded_grad = self.ret.grad.to("np").data.col2im(
                self.kernel_shape,
                self.input.shape[-2:],
                stride=self.stride,
            )
            self.input._grad += Tensor(folded_grad, storage=self.storage)


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
        folded = self.input.to("np").data.col2im(
            self.kernel_shape,
            self.output_shape,
            stride=self.stride,
        )
        self.ret = Tensor(
            folded,
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            unfolded = self.ret.grad.to("np").data.im2col(
                self.kernel_shape, stride=self.stride
            )
            self.input._grad += Tensor(unfolded, storage=self.storage)
