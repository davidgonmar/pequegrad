from .tensor import Tensor
from typing import List


class SGD:
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        self.params = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.vt_last = [0] * len(parameters)
        self.momentum = momentum

    def zero(self):
        for p in self.params:
            p.zero_grad()

    def step(self, zero: bool = True):
        for i, p in enumerate(self.params):
            gt = p.grad.data
            if self.weight_decay != 0:
                gt += self.weight_decay * p.data
            vt = self.momentum * self.vt_last[i] + gt
            self.vt_last[i] = vt
            p.data -= self.lr * vt

        if zero:
            self.zero()
