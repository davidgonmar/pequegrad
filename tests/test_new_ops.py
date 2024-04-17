import pequegrad.backend.c as pg
from pequegrad.backend.c import Tensor, dt, device 
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
        a = pg.fill((2, 3), dt.float32, 1, device.cpu)
        assert np.allclose(a.to_numpy(), np.ones((2, 3)))
    
    @pytest.mark.skip(reason="Not implemented")
    def test_custom(self):
        def torch_fn(a,b,c,d):
            return (a * b + c * d) * d
        
        def pq_fn(a,b,c,d):
            return pg.mul(pg.add(pg.mul(a, b), pg.mul(c, d)), d)
        
        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)
    
    def test_custom2(self):
        def torch_fn(a,b,c):
            return torch.mul(torch.add(torch.mul(a, b), c), b)
        
        def pq_fn(a,b,c):
            return pg.mul(pg.add(pg.mul(a, b), c), b)
        
        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)


    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @pytest.mark.parametrize("lambdaop", [
        (
            lambda x, y: pg.add(x, y),
            lambda x, y: torch.add(x, y),
            True
        ),
        (
            lambda x, y: pg.mul(x, y),
            lambda x, y: torch.mul(x, y),
            True
        ),
        (
            lambda x, y: pg.sub(x, y),
            lambda x, y: torch.sub(x, y),
            True
        ),
        (
            lambda x, y: pg.div(x, y),
            lambda x, y: torch.div(x, y),
            True
        ),
        (
            lambda x, y: pg.pow(x, y),
            lambda x, y: torch.pow(x, y),
            True
        ),
        (
            lambda x, y: pg.gt(x, y),
            lambda x, y: torch.gt(x, y),
            False
        ),
        (
            lambda x, y: pg.lt(x, y),
            lambda x, y: torch.lt(x, y),
            False
        ),
        (
            lambda x, y: pg.neq(x, y),
            lambda x, y: torch.ne(x, y),
            False
        ),
    ])
    def test_binary_ops(self, shape, dtype, lambdaop):
        pq_fn, torch_fn, do_backward = lambdaop
        _compare_fn_with_torch([shape, shape], pq_fn, torch_fn, backward=do_backward)
    

    # REDUCERS TESTS
    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @pytest.mark.parametrize("axes", [(0, 1), (1, 0), (0,), (1,), None])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("lambdaop", [
        (
            lambda x, axes, keepdims: pg.sum(x, axes, keepdims),
            lambda x, axes, keepdims: torch.sum(x, dim=axes, keepdim=keepdims),
            True
        ),
        (
            lambda x, axes, keepdims: pg.mean(x, axes, keepdims),
            lambda x, axes, keepdims: torch.mean(x, dim=axes, keepdim=keepdims),
            True
        ),
        (
            lambda x, axes, keepdims: pg.max_reduce(x, axes, keepdims),
            lambda x, axes, keepdims: torch.max(x, dim=axes, keepdim=keepdims)[0],
            True
        ),
    ])
    def test_reducers(self, shape, dtype, axes, keepdims, lambdaop):
        def pq_fn(x):
            return pg.sum(x, axes, keepdims)
        def torch_fn(x):
            return torch.sum(x, dim=axes, keepdim=keepdims)
        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=lambdaop[2])


    # Test broadcast to
    @pytest.mark.parametrize("shapes", [
        ((2, 3), (4, 2, 3)),
        ((2, 3), (2, 3)),
        ((2, 3), (1, 2, 3)),
        ((1, 2, 3), (2, 2, 3)),
    ])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    def test_broadcast_to(self, shapes, dtype):
        shape, target_shape = shapes
        def pq_fn(x):
            return pg.broadcast_to(x, target_shape)
        def torch_fn(x):
            return torch.broadcast_to(x, target_shape)
        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True)


    # Test permute
    @pytest.mark.parametrize("shape_and_dims", [
        ((2, 3), (1, 0)),
        ((2, 3, 4), (2, 0, 1)),
        ((2, 3, 4), (0, 1, 2)),
    ])
    @pytest.mark.parametrize("dtype", [dt.int32, dt.float32, dt.float64])
    def test_permute(self, shape_and_dims, dtype):
        shape, dims = shape_and_dims
        def pq_fn(x):
            return pg.permute(x, dims)
        def torch_fn(x):
            return torch.permute(x, dims)
        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True)