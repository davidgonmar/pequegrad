from pequegrad import Tensor, device
from pequegrad.compile import jit
import numpy as np
import time

@jit
def some_function(x):
    return x.log().exp().log().exp()

def non_jitted(x):
    return x.log().exp().log().exp()

for i in range(10):
    x = Tensor(np.random.randn(1000, 1000), device=device.cuda)
    start = time.time()
    j = some_function(x).eval()
    jittedtime = time.time() - start

    start = time.time()
    nj = non_jitted(x).eval()
    nonjittedtime = time.time() - start

    print(f"Jitted time: {jittedtime}, Non-jitted time: {nonjittedtime}")
    np.testing.assert_allclose(j.numpy(), nj.numpy(), atol=1e-5)
