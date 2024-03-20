from .function import Function, BackendTensor
from typing import Tuple


class MatMul(Function):
    def forward(self, x: BackendTensor, y: BackendTensor) -> BackendTensor:
        ret = x @ y
        if self.requires_grad:
            self.x = x
            self.y = y
        return ret

    def backward(
        self, grad_output: BackendTensor
    ) -> Tuple[BackendTensor, BackendTensor]:
        # If we are multiplying 2 matrices, we need to check out cases where one of them is batched
        # ,of shape (...extra_dims, m, n) and the other is not (of shape (n, k)).
        # In that case, we need to sum over the batch dimensions to get the gradient of the non-batched matrix
        # so the gradient has the same shape as the non-batched matrix
        x_grad, y_grad = None, None
        if self.needs_input_grad[0]:
            if self.x.ndim == 1 and self.y.ndim == 1:
                # Just multiply the gradients if both are vectors, since grad is a scalar
                x_grad = grad_output * self.y
            elif self.x.ndim == 1:
                # Vector x Matrix
                x_grad = grad_output @ self.y.T

            elif self.y.ndim == 1:
                # Matrix x Vector
                diff = self.y.ndim - self.x.ndim
                axis_to_sum_over = tuple(range(diff))
                x_grad = grad_output.outer_product(self.y).sum(axis=axis_to_sum_over)
            else:
                # Matrix x Matrix
                diff = self.y.ndim - self.x.ndim
                axis_to_sum_over = tuple(range(diff))
                x_grad = (grad_output @ self.y.swapaxes(-1, -2)).sum(
                    axis=axis_to_sum_over
                )

        if self.needs_input_grad[1]:
            if self.x.ndim == 1 and self.y.ndim == 1:
                y_grad = self.x * grad_output
            elif self.x.ndim == 1:
                y_grad = self.x.outer_product(grad_output)
            elif self.y.ndim == 1:
                # Matrix x Vector
                y_grad = grad_output.T @ self.x
            else:
                # Matrix x Matrix
                diff = self.x.ndim - self.y.ndim
                axis_to_sum_over = tuple(range(diff))
                y_grad = (self.x.swapaxes(-1, -2) @ grad_output).sum(
                    axis=axis_to_sum_over
                )

        return x_grad, y_grad
