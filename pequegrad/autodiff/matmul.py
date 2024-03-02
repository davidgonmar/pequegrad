from pequegrad.tensor import Tensor
from .function import Function


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
