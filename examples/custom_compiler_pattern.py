from pequegrad import Tensor, np, device, extend, add_custom_pattern
import pequegrad as pg
import torch


def numpy_add_pattern(tensor):
    # check if prim is Add
    if tensor.ad_context() == "Add":
        return tensor.children()
    return []


def numpy_add_pattern_converter(inps):
    class NumpyAdd(extend.Primitive):
        @staticmethod
        def dispatch(inputs):
            inps = [i.numpy() for i in inputs]
            inps = [torch.tensor(i) for i in inps]
            return (
                Tensor((inps[0] + inps[1]).numpy())
                .to(inputs[0].device)
                .astype(inputs[0].dtype)
            )

        @staticmethod
        def precompute(inputs):
            return [inputs[0]]

    return NumpyAdd.apply(*inps)


add_custom_pattern("numpy_add", numpy_add_pattern, numpy_add_pattern_converter)


@pg.jit.withargs(eval_outs=False)
def f(x, y):
    return x * 2 + y * 3


cuda = device.cuda(0)
shape = (16, 16)
x = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
y = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
res = f(x, y)
pg.viz(res, name="custom_compiler_pattern")
f.print_trace()

print(res.numpy())
