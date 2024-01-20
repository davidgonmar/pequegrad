from .tensor import Tensor
from typing import List


class SGD:
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.1,
        weight_decay: float = 0.0,
    ):
        self.params = parameters
        self.lr = lr
        self.weight_decay = weight_decay

    def zero(self):
        for p in self.params:
            p.zero_grad()

    def step(self, zero: bool = True):
        for p in self.params:
            p.data -= (p.grad.data + self.weight_decay * p.data) * self.lr
        if zero:
            self.zero()
