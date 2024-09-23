from pequegrad.linalg import lu_factorization
from pequegrad import Tensor, device
import numpy as np

print("LU factorization")
A = np.array([[2, 3, 1], [4, 7, 2], [6, 18, 5]], dtype=float)
print("Original matrix A:")
A = Tensor(A)
print(A.numpy())

L, U = lu_factorization(A)

print("\nFactorized matrix (LU combined):")
print(L.numpy(), "\n", U.numpy())
print((L @ U).numpy())
print((L @ U - A).numpy())


# speed test vs torch on cuda

import time
from pequegrad import jit
import torch

A = np.random.rand(100, 100).astype(np.float32)
t = Tensor(A).to(device.cuda)
torcht = torch.tensor(A).cuda()
jitted = jit(lu_factorization, opts={"common_subexpr_elim": False})
# warmup
r = jitted(t)
r[0].eval()
r[1].eval()


t0 = time.time()
for i in range(100):
    r = jitted(t)
    r[0].eval()
    r[1].eval()
t1 = time.time()

print(f"Pequegrad: {t1 - t0:.4f} seconds")

t0 = time.time()
for i in range(100):
    torch.lu(torcht)
t1 = time.time()

print(f"Torch: {t1 - t0:.4f} seconds")
