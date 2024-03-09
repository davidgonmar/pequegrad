from .function import Function
from ..tensor import Tensor
from typing import List, Tuple, Union


class Slice(Function):
    def __init__(self, x: Tensor, key: Union[int, slice, List[int], Tuple[int]]):
        super().__init__(x)
        self.x = x
        self.key = key

    def forward(self):
        self.ret = Tensor(
            self.x.data[self.key],
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.x.requires_grad:
            g = Tensor.zeros(self.x.shape, requires_grad=False, storage=self.storage)

            g[self.key] = self.ret.grad.data

            self.x._grad += g
