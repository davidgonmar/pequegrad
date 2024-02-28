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

    def reset_grad(self):
        for p in self.params:
            p.reset_grad()

    def step(self, reset_grad: bool = True):
        for i, p in enumerate(self.params):
            gt = p.grad.data
            if self.weight_decay != 0:
                gt += self.weight_decay * p.data
            vt = self.momentum * self.vt_last[i] + gt
            self.vt_last[i] = vt
            p.assign(p.data - self.lr * vt)

        if reset_grad:
            self.reset_grad()


class Adam:
    def __init__(
        self, parameters: List[Tensor], lr: float = 0.001, b1=0.9, b2=0.999, eps=1e-08
    ):
        self.params = parameters
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.mt_last = [0] * len(parameters)
        self.vt_last = [0] * len(parameters)
        self.t = 1
        self.eps = eps

    def reset_grad(self):
        for p in self.params:
            p.reset_grad()

    def step(self, reset_grad: bool = True):
        for i, p in enumerate(self.params):
            gt = p.grad.data
            mt = self.b1 * self.mt_last[i] + (1 - self.b1) * gt
            vt = self.b2 * self.vt_last[i] + (1 - self.b2) * (gt * gt)
            mt_hat = mt / (1 - self.b1**self.t)
            vt_hat = vt / (1 - self.b2**self.t)
            self.vt_last[i] = vt
            self.mt_last[i] = mt
            p.assign(p.data - self.lr * mt_hat / (vt_hat**0.5 + self.eps))
            self.t += 1
        if reset_grad:
            self.reset_grad()
