from .autodiff import fngrad, Tensor
from typing import Callable


def taylor_expand(fun: Callable, order: int) -> Callable:
    def taylor_fun(at: Tensor, x: Tensor) -> Tensor:
        res = fun(at)  # f(a)
        currfact = 1
        curr_dfdx = fngrad(fun, wrt=[0])
        for i in range(1, order + 1):
            term = (curr_dfdx(at)[0] * (x - at) ** i) / currfact
            res += term
            currfact *= i + 1
            curr_dfdx = fngrad(curr_dfdx, wrt=[0])
        return res

    return taylor_fun
