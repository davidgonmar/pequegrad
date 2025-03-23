import matplotlib.pyplot as plt
import numpy as np
from pequegrad.transforms import taylor_expand, jit, Tensor
import math
import time


def f(x):
    return (x + 1).log() + 3 * x**2 - 2 * x + 5 + math.e**x


# TODO -- jvp creates too much variables. See if we can make it more efficient
# + there is a fuser bug that emerges here
f_taylor_nojit = taylor_expand(f, order=3)
f_taylor = jit.withargs(eval_outs=False, opts={"fuser": False})(
    taylor_expand(f, order=3)
)  # No fuser so we can see the ops

# count time to eval jit vs no jit
x = Tensor(1.0).to("cuda")
x0 = Tensor(2.0).to("cuda")

# warmup jitted
f_taylor(x0, x).numpy()


print(f"Original function: {f(x).numpy()}")
# time no jit
start = time.time()
for i in range(10):
    f_taylor_nojit(x0, x).numpy()
end = time.time()
print(f"Time taken without jit: {end - start:.4f} s")

# time jit
start = time.time()
for i in range(10):
    f_taylor(x0, x).numpy()
end = time.time()
print(f"Time taken with jit: {end - start:.4f} s")

x_vals = np.linspace(0, 3, 100)
f_vals = np.array([f(Tensor(xi)).numpy() for xi in x_vals])
f_taylor_vals = np.array([f_taylor(x0, Tensor(xi).to("cuda")).numpy() for xi in x_vals])


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


# try actual tensors
v = Tensor([1.0, 2.0, 3.0]).to("cuda")
M = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).to("cuda")


def f(x):
    return (x @ M) + v


f_taylor = taylor_expand(f, order=3)

x = Tensor([1.0, 2.0, 3.0]).to("cuda")

print(f"Original function: {f(x).numpy()}")

x0 = Tensor([2.0, 3.0, 4.0]).to("cuda")

print(
    f"Taylor expansion: {f_taylor(x0, x).numpy()}"
)  # should be exactly the same as f(x0) since it is an affine function

# this does not work for quadratic functions (throws tangent shape mismatch -- SEE)
