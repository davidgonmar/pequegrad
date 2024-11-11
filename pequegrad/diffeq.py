from typing import Callable
from pequegrad.tensor import Tensor
from pequegrad.ops import cat


# y_t = y_{t-1} + f(y_{t-1}, t) * h, where f(y_{t-1}, t) represents dy/dt
def _euler_step(func: Callable, y: Tensor, t: Tensor, dt: Tensor) -> Tensor:
    return y + dt * func(y, t)


_steps = {"euler": _euler_step}


def odeint(
    func: Callable,
    y0: Tensor,
    t0: Tensor,
    dt: Tensor | float,
    num_steps: int,
    solver="euler",
) -> Tensor:
    assert y0.ndim == 0, t0.ndim == 0
    ys = [y0]
    ts = [t0]
    solver = _steps[solver]
    for _ in range(num_steps):
        ys.append(solver(func, ys[-1], ts[-1], dt))
    return cat(ys)
