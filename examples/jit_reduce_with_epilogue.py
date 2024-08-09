import pequegrad as pg
from pequegrad.compile import jit
import numpy as np
import pequegrad.viz as viz


x = (
    pg.Tensor(np.random.randn(10, 11, 6))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)
y = (
    pg.Tensor(np.random.randn(10, 11))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)
z = (
    pg.Tensor(np.random.randn(10, 11))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)


def reduce_log(x, y, z):
    summed0 = x.sum(dim=[0, 2])  # shape (11,)
    summed1 = x.sum(dim=2)  # shape (10, 11)
    out = summed1 * -1.0 + y * z  # shape (10, 11)
    return (out + summed0) @ z.T  # shape (10, 11)


out = jit(reduce_log)(x, y, z)
print("ooooo", out)
out.eval(False)
viz.viz(out, name="graph")
res = out.numpy()

resnojit = reduce_log(x, y, z)
print("resnojiiiiit", resnojit)
resnojit.eval(False)
resnojit = resnojit.numpy()

np.testing.assert_allclose(res, resnojit, rtol=1e-3)
