from pequegrad import Tensor, device, dt
from pequegrad.compile import jit
import numpy as np
import time


@jit
def some_function(x, y, z):
    return x.log() + y + z.exp().log().exp()


def non_jitted(x, y, z):
    return x.log() + y + z.exp().log().exp()


for i in range(10):
    x = Tensor(np.random.randn(10000, 1000), device=device.cuda).astype(dt.float32)
    y = Tensor(np.random.randn(10000, 1000), device=device.cuda).astype(dt.float32)
    z = Tensor(np.random.randn(10000, 1000), device=device.cuda).astype(dt.float32)

    start = time.time()
    nj = non_jitted(x, y, z).eval()
    nonjittedtime = time.time() - start

    start = time.time()
    j = some_function(x, y, z).eval()
    jittedtime = time.time() - start

    print(f"Jitted time: {jittedtime}, Non-jitted time: {nonjittedtime}")
    np.testing.assert_allclose(j.numpy(), nj.numpy(), atol=1e-5)
