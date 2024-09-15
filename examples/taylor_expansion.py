import matplotlib.pyplot as plt
import numpy as np
from pequegrad.transforms import taylor_expand, jit, Tensor
import math
import time


def f(x):
    return (x + 1).log() + 3 * x**2 - 2 * x + 5 + math.e**x


f_taylor_nojit = taylor_expand(f, order=5)
f_taylor = jit.withargs(opts={"fuser": False})(
    taylor_expand(f, order=5)
)  # No fuser so we can see the ops

# count time to eval jit vs no jit
x = Tensor(1.0)
x0 = Tensor(2.0)

# warmup jitted
f_taylor(x0, x).numpy()

# time no jit
start = time.time()
for i in range(100):
    f_taylor_nojit(x0, x).numpy()
end = time.time()
print(f"Time taken without jit: {end - start:.4f} s")

# time jit
start = time.time()
for i in range(100):
    f_taylor(x0, x).numpy()
end = time.time()
print(f"Time taken with jit: {end - start:.4f} s")

x_vals = np.linspace(0, 3, 100)
f_vals = np.array([f(Tensor(xi)).numpy() for xi in x_vals])
f_taylor_vals = np.array([f_taylor(x0, Tensor(xi)).numpy() for xi in x_vals])


plt.figure(figsize=(8, 6))
plt.plot(x_vals, f_vals, label="Original function", color="blue")
plt.plot(
    x_vals,
    f_taylor_vals,
    label="Taylor expansion (order 4)",
    color="red",
    linestyle="dashed",
)


plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Original function vs Taylor Expansion")
plt.legend()

plt.grid(True)
plt.show()
