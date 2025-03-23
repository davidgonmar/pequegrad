from .autodiff import jvp, Tensor
from typing import Callable
import pequegrad.ops as ops
import functools as ft


def partial_last(func, *args_fixed):
    @ft.wraps(func)
    def wrapper(*args_dynamic, **kwargs):
        return func(*args_dynamic, *args_fixed, **kwargs)

    return wrapper


def taylor_expand(fun: Callable, order: int) -> Callable:
    def taylor_fun(at: Tensor, x: Tensor) -> Tensor:
        res = fun(at)  # f(a)
        currfact = 1
        curr_dfdx = jvp(fun, wrt=[0])
        diff_param = x - at
        for i in range(1, order + 1):
            term = (curr_dfdx(at, diff_param)[0]) / currfact
            res += term
            currfact *= i + 1
            curr_dfdx = jvp(partial_last(curr_dfdx, diff_param), wrt=[0])
        return res

    return taylor_fun


def montecarlo_integrate(fun: Callable, n_samples: int) -> Callable:
    # returns a func that takes a and b and returns the integral of fun from a to b
    def montecarlo_integrate_fun(a: Tensor, b: Tensor) -> Tensor:
        # generates random samples from the selected interval, then computes the mean of the function
        x = ops.rand((n_samples,), a.dtype, a.device)  # random samples
        x = a + (b - a) * x
        return (b - a) * fun(x).mean()

    return montecarlo_integrate_fun
