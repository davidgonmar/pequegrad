from pequegrad import fngrad, fnjacobian, Tensor, dt
import numpy as np
import torch


def f(a, b, c):
    x = a * b
    return x * c


a, b, c = (
    Tensor(np.random.rand(5, 5)),
    Tensor(np.random.rand(5, 5)),
    Tensor(np.random.rand(5, 5)),
)
a, b, c = a.astype(dt.float32), b.astype(dt.float32), c.astype(dt.float32)

at, bt, ct = (
    torch.tensor(a.numpy(), requires_grad=True),
    torch.tensor(b.numpy(), requires_grad=True),
    torch.tensor(c.numpy(), requires_grad=True),
)

# GRADIENT

f_and_grad = fngrad(f, wrt=[0, 1, 2], return_outs=True)
res, grads = f_and_grad(a, b, c)

torch_res, torch_vjpfunc = torch.func.vjp(f, at, bt, ct)

torch_grads = torch_vjpfunc(torch.tensor(np.ones_like(res.numpy())))

for i in range(3):
    np.testing.assert_allclose(
        grads[i].numpy(), torch_grads[i].detach().numpy(), rtol=1e-5
    )


# JACOBIAN

f_and_jacobian = fnjacobian(f, wrt=[0, 1, 2], return_outs=True)

res, jacobian = f_and_jacobian(a, b, c)

torch_jacobian = torch.func.jacrev(f, argnums=(0, 1, 2))(at, bt, ct)

for i in range(3):
    np.testing.assert_allclose(
        jacobian[i].numpy(), torch_jacobian[i].detach().numpy(), rtol=1e-5
    )
