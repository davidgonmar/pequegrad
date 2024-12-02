from pequegrad import Tensor, np, device, extend, make_pattern_matcher, make_pattern
import pequegrad as pg
import torch


def cross_entropy_loss_pattern_fn(logits, probs):
    return pg.cross_entropy_loss_probs(logits, probs)


cross_entropy_loss_pattern = make_pattern_matcher(
    cross_entropy_loss_pattern_fn, [(16, 16), (16, 16)]
)


class TorchCrossEntropy(extend.Primitive):
    @staticmethod
    def dispatch(inputs):
        logits, probs = inputs
        torcht = torch.nn.functional.cross_entropy(
            torch.tensor(logits.numpy()), torch.tensor(probs.numpy()), reduction="mean"
        )
        return Tensor(torcht.numpy()).astype(pg.dt.float32).to(logits.device)

    @staticmethod
    def precompute(inputs):
        # Returns a tensor with the shape, dtype and device of the output. Contents don't matter.
        return [Tensor(1.0).astype(pg.dt.float32).to(inputs[0].device)]


def cross_entropy_loss_pattern_converter(inps):
    return TorchCrossEntropy.apply(*inps)


pat = make_pattern(
    "cross_entropy_loss_torch",
    cross_entropy_loss_pattern,
    cross_entropy_loss_pattern_converter,
    True,
)


def f(x, y):
    x = (x + 2) + 3 * x
    y = (y * y) + 2 + y
    return pg.cross_entropy_loss_probs(x, y) * 10


fjitted = pg.jit(f, eval_outs=False, custom_patterns=[pat])

cuda = device.cuda(0)
shape = (16,)
x = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
y = Tensor(np.random.rand(*shape), device=cuda).astype(pg.dt.float32)
res = fjitted(x, y)
pg.viz(res, name="custom_compiler_pattern")
fjitted.print_trace()

print(res.numpy())

print(f(x, y).numpy())
