from typing import Optional
from pequegrad.tensor import Tensor
from pequegrad.function import Function


class ReduceFunction(Function):
    def __init__(self, a: Tensor, dim: Optional[int] = None, keepdim: bool = False):
        super().__init__(a)
        self.a = a
        self.dim = dim
        self.keepdim = keepdim
    
    def _unreduce(self, x):
        # When keepdim is False, we need to insert a dimension of size 1 at the dimension we reduced over
        # so that broadcasting works correctly during the backward pass.
        if not self.keepdim and self.dim is not None:
            x = x.expand_dims(axis=self.dim)
        # Now we can broadcast the gradient to the shape of the input tensor
        return x.broadcast_to(self.a.shape)
    
class Mean(ReduceFunction):
    def forward(self):
        self.ret = Tensor(
            self.a.data.mean(axis=self.dim, keepdims=self.keepdim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            grad_broadcasted = self._unreduce(self.ret.grad.data)
            # Divide the gradient by the number of elements WE SUMMED OVER(not all elements)
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


class Sum(ReduceFunction):
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
            grad_broadcasted = self._unreduce(self.ret.grad.data)
            self.a._grad += Tensor(grad_broadcasted, storage=self.storage)


class Max(ReduceFunction):
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
            grad_broadcasted = self._unreduce(self.ret.grad.data)
            ret_broadcasted = self._unreduce(self.ret.data)

            self.a._grad += Tensor(
                grad_broadcasted.where(self.a.data == ret_broadcasted, 0),
                storage=self.storage,
            )


