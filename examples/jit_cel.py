import pequegrad as pg
from pequegrad.compile import jit
import numpy as np
import pequegrad.viz as viz


x = (
    pg.Tensor(np.random.randn(10, 11, 5))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)
y = (
    pg.Tensor(np.random.randn(10, 11, 5))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)


def crossentropy(x, y):
    return x.cross_entropy_loss_probs(y)


out = jit(crossentropy)(x, y)
print("ooooo", out)
out.eval(False)
viz.viz(out, name="graph")
res = out.numpy()

resnojit = crossentropy(x, y)
print("resnojiiiiit", resnojit)
resnojit.eval(False)
resnojit = resnojit.numpy()

np.testing.assert_allclose(res, resnojit, rtol=1e-3)
