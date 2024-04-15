from pequegrad.backend.c import *
import numpy as np
import torch
from torch import tensor as torch_tensor, Tensor as TorchTensor
import pytest



def _compare_fn_with_torch(
    shapes,
    pequegrad_fn,
    torch_fn=None,
    tol: float = 1e-5,
    backward=True,
):
    # In cases where the api is the same, just use the same fn as pequegrad
    torch_fn = torch_fn or pequegrad_fn

    # Ensure deterministic results
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Use a uniform distribution to initialize the arrays with 'good numbers' so that there are no numerical stability issues
    np_arr = [np.random.uniform(low=0.5, high=0.9, size=shape) for shape in shapes]
    tensors = [
        Tensor.from_numpy(arr.astype(np.float64))
        for arr in np_arr
    ]  # Using double precision

    torch_tensors = [
        torch_tensor(arr, dtype=torch.float64, requires_grad=True) for arr in np_arr
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

    _compare(peq_res, torch_res, tol)

    if backward:
        # Do it with 2 to ensure previous results are taken into account (chain rule is applied correctly)
        nparr = np.random.uniform(low=0.5, high=0.9, size=peq_res.shape)
        peq_res.backward(Tensor.from_numpy(nparr.astype(np.float64)))
        torch_res.backward(torch_tensor(nparr, dtype=torch.float64))

        for i, (t, torch_t) in enumerate(zip(tensors, torch_tensors)):
            print("Comparing position: ", i)
            _compare(t.grad, torch_t.grad, tol)



class TestNew:
    def test_fill(self):
        a = fill((2, 3), dt.float32, 1)
        assert np.allclose(a.to_numpy(), np.ones((2, 3)))
    
    @pytest.mark.skip(reason="Not implemented")
    def test_custom(self):
        def torch_fn(a,b,c,d):
            return (a * b + c * d) * d
        
        def pq_fn(a,b,c,d):
            return mul(add(mul(a, b), mul(c, d)), d)
        
        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)
    
    def test_custom2(self):
        def torch_fn(a,b,c):
            return torch.mul(torch.add(torch.mul(a, b), c), b)
        
        def pq_fn(a,b,c):
            return mul(add(mul(a, b), c), b)
        
        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)


    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @pytest.mark.parametrize("lambdaop", [
        (
            lambda x, y: add(x, y),
            lambda x, y: torch.add(x, y),
        ),
        (
            lambda x, y: mul(x, y),
            lambda x, y: torch.mul(x, y),
        ),
        (
            lambda x, y: sub(x, y),
            lambda x, y: torch.sub(x, y),
        ),
        (
            lambda x, y: div(x, y),
            lambda x, y: torch.div(x, y),
        ),
    ])
    def test_binary_ops(self, shape, dtype, lambdaop):
        pq_fn, torch_fn = lambdaop
        _compare_fn_with_torch([shape, shape], pq_fn, torch_fn)
                                          