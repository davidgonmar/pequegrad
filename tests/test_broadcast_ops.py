from pequegrad.backend.c import Tensor, dt, device, grads
import pequegrad.backend.c as pg
import numpy as np
import torch
from torch import tensor as torch_tensor, Tensor as TorchTensor
import pytest  # noqa

dtypemapnp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}
dtypemapt = {
    dt.float32: torch.float32,
    dt.float64: torch.float64,
    dt.int32: torch.int32,
}


def _compare_fn_with_torch(
    shapes,
    pequegrad_fn,
    torch_fn=None,
    tol: float = 1e-5,
    backward=True,
    device: device = device.cpu,
    dtype=dt.float64,
):
    # In cases where the api is the same, just use the same fn as pequegrad
    torch_fn = torch_fn or pequegrad_fn

    # Ensure deterministic results
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Use a uniform distribution to initialize the arrays with 'good numbers' so that there are no numerical stability issues
    np_arr = (
        [np.random.uniform(low=0.5, high=0.9, size=shape) for shape in shapes]
        if dtype in [dt.float32, dt.float64]
        else [np.random.randint(1, 4, size=shape) for shape in shapes]
    )
    tensors = [
        Tensor.from_numpy(arr.astype(dtypemapnp[dtype])).to(device) for arr in np_arr
    ]  # Using double precision

    torch_tensors = [
        torch_tensor(arr, dtype=dtypemapt[dtype], requires_grad=backward)
        for arr in np_arr
    ]  # Using double precision

    torch_res = torch_fn(*torch_tensors)
    peq_res = pequegrad_fn(*tensors)

    def _compare(t: Tensor, torch_t: TorchTensor, tol: float = 1e-5):
        list1 = np.array(t.numpy()) if t is not None else None
        list2 = np.array(torch_t.detach().numpy()) if torch_t is not None else None

        assert type(list1) == type(list2)

        assert list(t.shape) == list(
            torch_t.shape
        ), f"t.shape: {t.shape} != torch_t.shape: {torch_t.shape}"

        np.testing.assert_allclose(list1, list2, rtol=tol, atol=tol)

    if backward:
        nparr = np.random.uniform(low=0.5, high=0.9, size=peq_res.shape)
        peq_grads = grads(
            tensors,
            peq_res,
            Tensor.from_numpy(nparr.astype(dtypemapnp[dtype])).to(device),
        )
        torch_res.backward(torch_tensor(nparr, dtype=dtypemapt[dtype]))
        torch_grads = [t.grad for t in torch_tensors]
        assert len(peq_grads) == len(torch_grads)
        for i, (t, torch_t) in enumerate(zip(peq_grads, torch_grads)):
            print("Comparing position: ", i)
            _compare(t, torch_t, tol)
    _compare(peq_res, torch_res, tol)


class TestBroadcastOps:
    def test_add(self):
        shapes = [(10, 20), (20)]
        _compare_fn_with_torch(shapes, pg.add, torch.add)
