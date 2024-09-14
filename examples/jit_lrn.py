import pequegrad as pg
from pequegrad.transforms.compile import jit
import numpy as np
import pequegrad.viz as viz
import torch


# x has shape (batch_size, in_channels, height, width)
x = (
    pg.Tensor(np.random.randn(1, 2, 100, 100))
    .to(pg.device.cuda)
    .astype(pg.dt.float32)
    .eval()
    .detach()
)

torchx = torch.tensor(x.numpy(), requires_grad=True)


def lrn(x):
    out = pg.local_response_norm(x, size=5, k=2, alpha=1e-4, beta=0.75)
    g = pg.grads([x], out, pg.fill(out.shape, pg.dt.float32, 12222.0, pg.device.cuda))
    return out + 2, g[0] + 2


def torchlrn(x):
    out = torch.nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)(x)
    out.backward(torch.ones_like(out))
    return out + 2, x.grad + 2


out = jit(lrn)(x)
viz.viz(out, name="graph")
outnojit = lrn(x)
outtorch = torchlrn(torchx)
out[1].eval(False)

x = out[1].children()


outnp = out[0].numpy()
outgnp = out[1].numpy()
outnojitnp = outnojit[0].numpy()
outnojitgnp = outnojit[1].numpy()
outtorchnp = outtorch[0].detach().numpy()
outtorchgnp = outtorch[1].detach().numpy()

# print(outgnp)
# print(outtorchgnp)
