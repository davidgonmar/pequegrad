import pequegrad as pg
from pequegrad.compile import jit
import numpy as np
import pequegrad.viz as viz


# x has shape (batch_size, in_channels, height, width)
x = (
    pg.Tensor(np.random.randn(11, 6, 35, 35))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)


def pool_and_grad(x):
    out = pg.max_pool2d(x, kernel_size=(2, 2))
    g = pg.grads([x], out)
    return out, *g


out = jit(pool_and_grad)(x)
viz.viz(out, name="graph")
out[0].numpy()
out[1].numpy()
