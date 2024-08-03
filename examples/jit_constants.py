import pequegrad as pg
from pequegrad.compile import jit
import pequegrad.viz as viz
import numpy as np


x = pg.broadcast_to(pg.fill((), pg.dt.float32, 0.0, pg.device.cuda), (10,))
y = pg.broadcast_to(pg.fill((), pg.dt.float32, 1.0, pg.device.cuda), (10,))
z = (
    pg.Tensor(np.random.randn(10))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)


def fn(t):
    return (x + y).log() - t.exp()


out = jit(fn)(z)
viz.viz(out, name="graph")
out[0].numpy()
