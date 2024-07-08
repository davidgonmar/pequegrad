from pequegrad import custom_prim, Tensor, grads, device, dt
import numpy as np
from functools import partial


@partial(custom_prim, compile_jit=True)
def myfunction(x, y):
    return x + y


@myfunction.vjp
def myfunction_vjp(primals, tangents, outputs):
    x, y = primals
    return x + y * 3, y * tangents[0] + outputs[0]


a = Tensor(np.array([1.0, 2.0, 3.0])).to(device.cuda).astype(dt.float32)
b = Tensor(np.array([2.0, 3.0, 4.0])).to(device.cuda).astype(dt.float32)


c = myfunction(a, b)
g = grads([a, b], c)  # tangents implicitly set to 1.0

print(c.numpy())  # [3.0, 5.0, 7.0]
print(g[0].numpy())  # [7.0, 11.0, 15.0]
print(g[1].numpy())  # [5.0, 8.0, 11.0]
