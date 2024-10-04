from pequegrad.backend.c import Tensor
from typing import List
from pequegrad.transforms.compile import jit
from pequegrad.transforms.pytree import first_tensor_pytree, tree_map


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


class JittedSGD:
    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        self.params = parameters
        self.device = parameters[0].device
        self.lr = lr
        self.weight_decay = weight_decay
        self.vt_last = (
            [Tensor.zeros(p.shape).to(self.device) for p in parameters]
            if momentum != 0
            else None
        )
        self.momentum = momentum

        self.jitted_steps = [
            jit(
                self.one_param_step,
            )
            for _ in range(len(parameters))
        ]

    def one_param_step(self, p, gt, vt):
        if self.weight_decay != 0:
            gt += self.weight_decay * p
        if self.momentum != 0:
            vt = self.momentum * vt + gt
        newp = p - self.lr * (vt if self.momentum != 0 else gt)
        return (vt, newp) if self.momentum != 0 else (newp,)

    def step(self, g):
        assert len(g) == len(self.params)
        for i, (p, gt) in enumerate(zip(self.params, g)):
            vt = self.vt_last[i]
            res = self.jitted_steps[i](p, gt, vt)
            if self.momentum != 0:
                vt, newp = res
                self.vt_last[i] = vt.eval().detach()
            else:
                newp = res[0]
            p.assign(newp)
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
    def __init__(self, parameters, lr: float = 0.001, b1=0.9, b2=0.999, eps=1e-08):
        self.params = (
            parameters.tree_flatten()
            if hasattr(parameters, "tree_flatten")
            else parameters
        )
        self.device = first_tensor_pytree(parameters).device
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.mt_last = tree_map(
            lambda p: Tensor.zeros(p.shape).to(self.device), parameters.tree_flatten()
        )
        self.vt_last = tree_map(
            lambda p: Tensor.zeros(p.shape).to(self.device), parameters.tree_flatten()
        )
        self.t = Tensor(1).to(self.device)
        self.eps = eps
        self.jitted_steps = tree_map(
            lambda _: jit(
                self.one_param_step,
            ),
            parameters,
        )

    def one_param_step(self, p, gt, mt, vt, t):
        mt = self.b1 * mt + (1 - self.b1) * gt
        vt = self.b2 * vt + (1 - self.b2) * (gt**gt)
        mt_hat = mt / (1 - self.b1**t)
        vt_hat = vt / (1 - self.b2**t)
        newp = p - self.lr * mt_hat / (vt_hat**0.5 + self.eps)
        return mt, vt, newp

    def step(self, grads):
        def _update(p_old, mt_old, vt_old, mt_new, vt_new, p_new):
            p_old.assign(p_new)
            mt_old.assign(mt_new)
            vt_old.assign(vt_new)
            self.t.assign(self.t + 1)

        tree_map(
            lambda p, gt, mt, vt, fn: _update(p, mt, vt, *fn(p, gt, mt, vt, self.t)),
            self.params,
            grads,
            self.mt_last,
            self.vt_last,
            self.jitted_steps,
        )


### Functional API


class OptimizerState:
    pass


class AdamState(OptimizerState):
    def _state_dict_for_flatten(self):
        return {
            "mt": self.mt,
            "vt": self.vt,
            "t": self.t,
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "params": self.params,
        }

    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-08, _state_dict=None):
        if _state_dict is None:
            self.mt = tree_map(lambda p: Tensor.zeros(p.shape).to(p.device), params)
            self.vt = tree_map(lambda p: Tensor.zeros(p.shape).to(p.device), params)
            self.t = Tensor(1).to(first_tensor_pytree(params).device).eval()
        else:
            self.mt = _state_dict["mt"]
            self.vt = _state_dict["vt"]
            self.t = _state_dict["t"]
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.params = params

    def numpy(self):
        return {
            "mt": tree_map(lambda mt: mt.numpy(), self.mt),
            "vt": tree_map(lambda vt: vt.numpy(), self.vt),
            "t": self.t.numpy(),
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "params": tree_map(lambda p: p.numpy(), self.params),
        }

    def _from_dict(d):
        return AdamState(
            d["params"],
            lr=d["lr"],
            b1=d["b1"],
            b2=d["b2"],
            eps=d["eps"],
            _state_dict={
                "mt": d["mt"],
                "vt": d["vt"],
                "t": d["t"],
            },
        )


def adam(params, grads, state: AdamState):
    mt = tree_map(lambda mt, gt: mt * state.b1 + (1 - state.b1) * gt, state.mt, grads)
    vt = tree_map(
        lambda vt, gt: vt * state.b2 + (1 - state.b2) * (gt * gt), state.vt, grads
    )
    mt_hat = tree_map(lambda mt: mt / (1 - state.b1**state.t), mt)
    vt_hat = tree_map(lambda vt: vt / (1 - state.b2**state.t), vt)
    new_params = tree_map(
        lambda p, mt_hat, vt_hat: p - state.lr * mt_hat / (vt_hat**0.5 + state.eps),
        params,
        mt_hat,
        vt_hat,
    )
    new_state = AdamState(
        new_params,
        lr=state.lr,
        b1=state.b1,
        b2=state.b2,
        eps=state.eps,
        _state_dict={
            "mt": mt,
            "vt": vt,
            "t": state.t,
        },
    )
    return new_state
