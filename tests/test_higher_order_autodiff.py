import pytest
import numpy as np
import torch
from pequegrad import fngrad, fnjacobian, Tensor, dt, fnhessian, device, checkpoint


def function1(a, b):
    return a * b


def function2(a, b):
    return a + b


def function3(a, b):
    return a**2 + b**2


cuda = device.cuda(0)


@pytest.fixture
def tensors():
    a = Tensor(np.random.rand(2, 2), device=cuda).astype(dt.float32)
    b = Tensor(np.random.rand(2, 2), device=cuda).astype(dt.float32)
    at = torch.tensor(a.numpy(), requires_grad=True)
    bt = torch.tensor(b.numpy(), requires_grad=True)
    return a, b, at, bt


@pytest.mark.parametrize("func", [function1, function2, function3])
def test_gradient(tensors, func):
    a, b, at, bt = tensors
    f_and_grad = fngrad(func, wrt=[0, 1], return_outs=True)
    res, grads = f_and_grad(a, b)
    torch_res, torch_vjpfunc = torch.func.vjp(func, at, bt)
    torch_grads = torch_vjpfunc(torch.tensor(np.ones_like(res.numpy())))
    for i in range(2):
        np.testing.assert_allclose(
            grads[i].numpy(), torch_grads[i].detach().numpy(), rtol=1e-5
        )


@pytest.mark.parametrize("func", [function1, function2, function3])
def test_jacobian(tensors, func):
    a, b, at, bt = tensors
    f_and_jacobian = fnjacobian(func, wrt=[0, 1], return_outs=True)
    res, jacobian = f_and_jacobian(a, b)
    torch_jacobian = torch.func.jacrev(func, argnums=(0, 1))(at, bt)
    for i in range(2):
        np.testing.assert_allclose(
            jacobian[i].numpy(), torch_jacobian[i].detach().numpy(), rtol=1e-5
        )


@pytest.mark.parametrize("func", [function1, function2, function3])
def test_hessian(tensors, func):
    a, b, at, bt = tensors
    f_and_hessian = fnhessian(func, wrt=[0, 1], return_outs=True)
    res, hessians = f_and_hessian(a, b)
    torch_hessian = torch.func.hessian(func, argnums=(0, 1))(at, bt)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(
                hessians[i][j].numpy(), torch_hessian[i][j].detach().numpy(), rtol=1e-5
            )


@pytest.mark.parametrize("func", [function1, function2, function3])
def test_checkpoint(tensors, func):
    checkpointed_func = checkpoint(func)
    a, b, *_ = tensors
    res = func(a, b)
    checkpointed_res = checkpointed_func(a, b)
    checkpointed_grads = fngrad(checkpointed_func, wrt=[0, 1])(a, b)
    grads = fngrad(func, wrt=[0, 1])(a, b)
    for i in range(2):
        np.testing.assert_allclose(
            checkpointed_grads[i].numpy(), grads[i].numpy(), rtol=1e-5
        )
    np.testing.assert_allclose(res.numpy(), checkpointed_res.numpy(), rtol=1e-5)
