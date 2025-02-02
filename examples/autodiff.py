from pequegrad import fngrad, fnjacobian, Tensor, dt, fnhessian, fnvhp
import numpy as np
import torch


def f(a, b, c):
    x = a * b
    return x * c


a, b, c = (
    Tensor(np.random.rand(5, 5), device="cuda"),
    Tensor(np.random.rand(5, 5), device="cuda"),
    Tensor(np.random.rand(5, 5), device="cuda"),
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

print("Gradient test passed")

# JACOBIAN

f_and_jacobian = fnjacobian(f, wrt=[0, 1, 2], return_outs=True)

res, jacobian = f_and_jacobian(a, b, c)

torch_jacobian = torch.func.jacrev(f, argnums=(0, 1, 2))(at, bt, ct)

for i in range(3):
    np.testing.assert_allclose(
        jacobian[i].numpy(), torch_jacobian[i].detach().numpy(), rtol=1e-5
    )

print("Jacobian test passed")

# HESSIAN (this one takes a while :( )

f_and_hessian = fnhessian(f, wrt=[0, 1, 2], return_outs=True)

res, hessians = f_and_hessian(a, b, c)

torch_hessian = torch.func.hessian(f, argnums=(0, 1, 2))(at, bt, ct)

for i in range(3):
    for j in range(3):
        np.testing.assert_allclose(
            hessians[i][j].numpy(), torch_hessian[i][j].detach().numpy(), rtol=1e-5
        )

print("Hessian test passed")

# VECTOR HESSIAN PRODUCT


# needs a scalar function
def f_scalar(a, b):
    x = a * b
    return x.sum()


v = Tensor(np.random.rand(5, 5), device="cuda").astype(dt.float32)
vtorch = torch.tensor(v.numpy(), requires_grad=True)

f_and_vhp = fnvhp(f_scalar, wrt=[0, 1], return_outs=True)

res, vhp = f_and_vhp(a, b, v)

torch_res, torch_vhp = torch.autograd.functional.vhp(
    f_scalar, (at, bt), (vtorch, vtorch)
)

for i in range(2):
    np.testing.assert_allclose(vhp[i].numpy(), torch_vhp[i].detach().numpy(), rtol=1e-5)

print("Vector Hessian Product test passed")
