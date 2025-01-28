import pequegrad as pg
from pequegrad.transforms.compile import jit
import numpy as np


x = (
    pg.Tensor(np.random.randn(5, 5, 5, 5))
    .to("cuda")
    .astype(pg.dt.float32)
    .eval()
    .detach()
)
y = (
    pg.Tensor(np.random.randn(5, 5, 5, 5))
    .to("cuda")
    .astype(pg.dt.float32)
    .eval()
    .detach()
)

z = (
    pg.Tensor(np.random.randn(5, 5, 5, 5))
    .to("cuda")
    .astype(pg.dt.float32)
    .eval()
    .detach()
)


def fn(x, y, z):
    return x @ y + z


fnj = jit(fn, use_cuda_graph=True)

out = fnj(x, y, z)
out2 = fnj(x, y, z)

print(out.numpy())

res2 = x.numpy() @ y.numpy() + z.numpy()

print(res2)
