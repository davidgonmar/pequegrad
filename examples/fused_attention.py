from pequegrad import Tensor, np, device
import pequegrad as pg


@pg.jit.withargs(opts={"fuser": False})
def f(q, k, v):
    return pg.scaled_dot_product_attention(q, k, v)


cuda = device.cuda
shape = (32, 64, 16, 16)
q = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
k = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
v = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)


res = f(q, k, v)

pg.viz(res, name="fused_attention")
f.print_trace()
print(res)
print(res.numpy())
