import pytest
from pequegrad.tensor import Tensor
from torch import Tensor as TorchTensor, tensor as torch_tensor
import numpy as np
import torch


def _compare_fn_with_torch(shapes, pequegrad_fn, torch_fn=None, tol: float = 1e-5):
    # In cases where the api is the same, just use the same fn as pequegrad
    torch_fn = torch_fn or pequegrad_fn

    # Ensure deterministic results
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Make sure they are positive by taking the absolute value
    np_arr = [np.abs(np.random.randn(*shape)) for shape in shapes]
    tensors = [Tensor(arr, requires_grad=True) for arr in np_arr]
    torch_tensors = [torch_tensor(arr, requires_grad=True) for arr in np_arr]

    peq_res = pequegrad_fn(*tensors)
    torch_res = torch_fn(*torch_tensors)

    def _compare(t: Tensor, torch_t: TorchTensor, tol: float = 1e-5):
        list1 = np.array(t.tolist()) if t is not None else None
        list2 = np.array(torch_t.tolist()) if torch_t is not None else None

        assert type(list1) == type(list2)

        assert (
            t.shape == torch_t.shape
        ), f"t.shape: {t.shape} != torch_t.shape: {torch_t.shape}"

        assert np.allclose(
            list1, list2, atol=tol, equal_nan=True
        ), f"t: {list1} != torch_t: {list2}"

    _compare(peq_res, torch_res, tol)

    # Do it with 2 to ensure previous results are taken into account (chain rule is applied correctly)
    peq_res.backward(Tensor(np.full(peq_res.shape, 2.0)))
    torch_res.backward(torch_tensor(np.full(torch_res.shape, 2.0)))

    for t, torch_t in zip(tensors, torch_tensors):
        _compare(t.grad, torch_t.grad, tol)


class TestReshape:
    shapes = [(2, 3), (3, 2), (6, 1), (1, 6), (6,), (1, 1, 6), (1, 2, 3), (2, 3, 1)]

    @pytest.mark.parametrize("shape", shapes)
    def test_reshape(self, shape):
        _compare_fn_with_torch(
            [shape],
            lambda x: x.reshape(shape),
            lambda x: x.reshape(shape),
        )


class TestPow:
    shapes = [
        [(3,), (3,)],
        [(2, 3), (2, 3)],
        [(1, 2, 3), (1, 2, 3)],
    ]

    @pytest.mark.parametrize("shapes", shapes)
    def test_pow(self, shapes):
        _compare_fn_with_torch(
            shapes,
            lambda x, y: x**y,
            lambda x, y: x**y,
        )


class TestAdd:
    shapes = [
        [(1, 2, 3), (1, 2, 3)],
        [(1, 2, 3), (1, 2, 1)],
        [(1, 2, 3), (1, 1, 3)],
    ]

    @pytest.mark.parametrize("shapes", shapes)
    def test_add(self, shapes):
        _compare_fn_with_torch(
            shapes,
            lambda x, y: x + y,
            lambda x, y: x + y,
        )


class TestExp:
    shapes = [
        [(3,)],
        [(2, 3)],
        [(1, 2, 3)],
    ]

    @pytest.mark.parametrize("shapes", shapes)
    def test_exp(self, shapes):
        _compare_fn_with_torch(
            shapes,
            lambda x: x.exp(),
            lambda x: x.exp(),
        )


class TestMSE:
    shapes = [
        [(3,)],
        [
            (
                3,
                2,
            )
        ],
    ]

    @pytest.mark.parametrize("shape", shapes)
    def test_mse(self, shape):
        def torch_fn(x, y):
            mse = torch.nn.MSELoss(reduction="mean")
            return mse(x, y)

        _compare_fn_with_torch(shape * 2, lambda x, y: x.mse_loss(y), torch_fn)


class TestMul:
    shapes = [
        [(1, 2, 3), (1, 2, 3)],
        [(1, 2, 3), (1, 2, 1)],
        [(1, 2, 3), (1, 1, 3)],
    ]

    @pytest.mark.parametrize("shapes", shapes)
    def test_mul(self, shapes):
        _compare_fn_with_torch(shapes, lambda x, y: x * y, lambda x, y: x * y)


class TestReLU:
    shapes = [
        [
            (1,),
        ],
        [
            (
                1,
                2,
            )
        ],
        [(1, 2, 3)],
    ]

    @pytest.mark.parametrize("shape", shapes)
    def test_relu(self, shape):
        _compare_fn_with_torch(shape, lambda x: x.relu())


class TestSum:
    # shape, dim
    data = [[(2, 3), 0], [(2, 3), (0, 1)]]

    @pytest.mark.parametrize("data", data)
    def test_sum(self, data):
        shape, dim = data
        _compare_fn_with_torch(
            [shape], lambda x: x.sum(dim=dim), lambda x: x.sum(dim=dim)
        )


class TestTranspose:
    # shape, dim0, dim1
    data = [
        [(1, 2, 3), 0, 1],
    ]

    @pytest.mark.parametrize("data", data)
    def test_transpose(self, data):
        shape, dim0, dim1 = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.transpose(dim0, dim1),
            lambda x: x.transpose(dim0, dim1),
        )


class TestMatMul:
    shapes = [
        [(3,), (3,)],
        [(2,), (2, 2)],
        [(2, 2), (2, 2)],
        [(4, 1), (1,)],
        [(4,), (4, 1)],
    ]

    @pytest.mark.parametrize("shapes", shapes)
    def test_matmul(self, shapes):
        _compare_fn_with_torch(shapes, lambda x, y: x @ y, lambda x, y: x @ y)
