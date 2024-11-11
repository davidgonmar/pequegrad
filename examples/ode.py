import matplotlib.pyplot as plt
from pequegrad import Tensor
from pequegrad.diffeq import odeint
import pequegrad.ops as ops


def func(y: Tensor, t: Tensor) -> Tensor:
    return -y


y0 = Tensor(1.0)
t0 = Tensor(0.0)
dt = 0.01
num_steps = 1000

y_values = odeint(func, y0, t0, dt, num_steps, solver="euler")
t_values = t0 + dt * ops.arange(0, num_steps + 1, 1, y_values.dtype, y_values.device)

plt.plot(t_values.numpy(), y_values.numpy(), label="Euler's Method")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Solution of dy/dt = -y using Euler's Method")
plt.legend()
plt.grid(True)
plt.show()
