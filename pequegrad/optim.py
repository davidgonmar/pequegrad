from .tensor import Tensor
from typing import List


class SGD:
    def __init__(self, parameters: List[Tensor], lr: float = 0.1):
        self.params = parameters
        self.lr = lr

    def zero(self):
        for p in self.params:
            p.zero_grad()

    def step(self, zero: bool = True):
        for p in self.params:
            p.data -= p.grad.data * self.lr
        if zero:
            self.zero()
