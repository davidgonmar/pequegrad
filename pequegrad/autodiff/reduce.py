from typing import Optional
from pequegrad.tensor import Tensor
from .function import Function, BackendTensor


class ReduceFunction(Function):
    def _unreduce(self, x):
        # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
        # so that broadcasting works correctly during the backward pass.
        if not self.keepdim and self.dim is not None:
            x = x.expand_dims(axis=self.dim)
        # Now we can broadcast the gradient to the shape of the input tensor
        return x.broadcast_to(self.a.shape)


class Mean(ReduceFunction):
    def forward(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        if self.requires_grad:
            self.a = a
            self.dim = dim
            self.keepdim = keepdim
        return a.mean(axis=dim, keepdims=keepdim)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            grad_broadcasted = self._unreduce(grad_output)
            # Divide the gradient by the number of elements WE SUMMED OVER(not all elements)
            total_els = 1
            if self.dim is None:
                total_els = self.a.size
            elif isinstance(self.dim, int):
                total_els = self.a.shape[self.dim]
            else:
                total_els = 1
                for d in self.dim:
                    total_els *= self.a.shape[d]
            return grad_broadcasted / total_els


class Sum(ReduceFunction):
    def forward(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        if self.requires_grad:
            self.keepdim = keepdim
            self.dim = dim
            self.a = a
        return a.sum(axis=dim, keepdims=keepdim)

    def backward(self, grad_output: BackendTensor):
        if self.needs_input_grad[0]:
            return self._unreduce(grad_output)


class Max(ReduceFunction):
    def forward(
        self, a: BackendTensor, dim: Optional[int] = None, keepdim: bool = False
    ) -> BackendTensor:
        if self.requires_grad:
            self.a = a
            self.dim = dim
            self.keepdim = keepdim
            self.ret = a.max(axis=dim, keepdims=keepdim)
        return self.ret if self.requires_grad else a.max(axis=dim, keepdims=keepdim)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            grad_broadcasted = self._unreduce(grad_output)
            ret_broadcasted = self._unreduce(self.ret)

            return (self.a == ret_broadcasted).where(grad_broadcasted, 0)
