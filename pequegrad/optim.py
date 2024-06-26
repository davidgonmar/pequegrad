from pequegrad.backend.c import Tensor
from typing import List
from pequegrad.compile import jit


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

    def step(self, g):
        assert len(g) == len(self.params)
        for i, p in enumerate(self.params):
            gt = g[i]
            if self.weight_decay != 0:
                gt += self.weight_decay * p
            vt = self.momentum * self.vt_last[i] + gt
            p.assign(p - self.lr * vt)
            self.vt_last[i] = vt.eval().detach()
            del gt

        del g


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

    def step(self, grads):
        assert len(grads) == len(self.params)
        for i, p in enumerate(self.params):
            gt = grads[i]
            mt = self.b1 * self.mt_last[i] + (1 - self.b1) * gt
            vt = self.b2 * self.vt_last[i] + (1 - self.b2) * (gt * gt)
            mt_hat = mt / (1 - self.b1**self.t)
            vt_hat = vt / (1 - self.b2**self.t)
            p.assign(p - self.lr * mt_hat / (vt_hat**0.5 + self.eps))
            self.vt_last[i] = vt.eval().detach()
            self.mt_last[i] = mt.eval().detach()
            del gt
        del grads
        self.t += 1


class JittedAdam:
    def __init__(
        self, parameters: List[Tensor], lr: float = 0.001, b1=0.9, b2=0.999, eps=1e-08
    ):
        self.params = parameters
        self.device = parameters[0].device
        self.lr = Tensor(lr).to(self.device)
        self.b1 = Tensor(b1).to(self.device)
        self.b2 = Tensor(b2).to(self.device)
        self.mt_last = [Tensor.zeros(p.shape).to(self.device) for p in parameters]
        self.vt_last = [Tensor.zeros(p.shape).to(self.device) for p in parameters]
        self.t = Tensor(1).to(self.device)
        self.eps = Tensor(eps).to(self.device)

        self.jitted_steps = [
            jit(
                self.one_param_step,
                externals=lambda: [self.b1, self.b2, self.lr, self.eps, self.t],
            )
            for _ in range(len(parameters))
        ]

    def one_param_step(self, p, gt, mt, vt):
        mt = self.b1 * mt + (1 - self.b1) * gt
        vt = self.b2 * vt + (1 - self.b2) * (gt * gt)
        mt_hat = mt / (1 - self.b1**self.t)
        vt_hat = vt / (1 - self.b2**self.t)
        newp = p - self.lr * mt_hat / (vt_hat**0.5 + self.eps)
        return mt, vt, newp

    def step(self, grads):
        assert len(grads) == len(self.params)
        for i, (p, gt) in enumerate(zip(self.params, grads)):
            mt = self.mt_last[i]
            vt = self.vt_last[i]
            mt, vt, newp = self.jitted_steps[i](p, gt, mt, vt)
            self.vt_last[i] = vt.eval().detach()
            self.mt_last[i] = mt.eval().detach()
            p.assign(newp)
        self.t = (self.t + 1).eval().detach()
        del grads
