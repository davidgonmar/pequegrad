from .function import Function, BackendTensor
from typing import List, Tuple, Union


class Slice(Function):
    def forward(
        self, x: BackendTensor, key: Union[int, slice, List[int], Tuple[int]]
    ) -> BackendTensor:
        if self.requires_grad:
            self.x = x
            self.key = key
        return x[key]

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            g = self.x.fill(self.x.shape, 0, dtype=self.x.dtype)
            g[self.key] = grad_output
            return g
        return None
