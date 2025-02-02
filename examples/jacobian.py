from pequegrad import fnjacobian
import numpy as np
import torch
from pequegrad import Tensor, dt
import pequegrad as pg
import time


def f(a, b):
    return a * b


a, b = (
    Tensor(np.random.rand(200), device="cuda"),
    Tensor(np.random.rand(200), device="cuda"),
)
a, b = a.astype(dt.float32), b.astype(dt.float32)

at, bt = (
    torch.tensor(a.numpy(), requires_grad=True).to("cuda"),
    torch.tensor(b.numpy(), requires_grad=True).to("cuda"),
)

f_and_jacobian = pg.jit(
    fnjacobian(f, wrt=[0, 1], return_outs=True),
    eval_outs=False,
    opts={"fuser": False},
)

res, jacobian = f_and_jacobian(a, b)
pg.viz(jacobian[0], "jacobian")
torch_jacobian = torch.func.jacrev(f, argnums=(0, 1))(at, bt)
f_and_jacobian.print_trace()
for i in range(2):
    np.testing.assert_allclose(
        jacobian[i].numpy(), torch_jacobian[i].cpu().detach().numpy(), rtol=1e-5
    )

print("Jacobian test passed")


def time_function(func, *args, warmup=2, iterations=10):
    for _ in range(warmup):
        res = func(*args)
    start_time = time.time()
    for _ in range(iterations):
        res = func(*args)
    end_time = time.time()
    return (end_time - start_time) / iterations


non_jitted_f_and_jacobian = fnjacobian(f, wrt=[0, 1])
pg_no_jit_time = time_function(non_jitted_f_and_jacobian, a, b)
print(f"Pequegrad non-jitted average time: {pg_no_jit_time:.6f} seconds")

pg_jit_time = time_function(pg.jit(fnjacobian(f, wrt=[0, 1])), a, b)
print(f"Pequegrad jitted average time: {pg_jit_time:.6f} seconds")

torch_jacobian_func = lambda: torch.func.jacrev(f, argnums=(0, 1))(at, bt)
torch_time = time_function(torch_jacobian_func)
print(f"Torch average time: {torch_time:.6f} seconds")
