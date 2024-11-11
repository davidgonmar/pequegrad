import pequegrad as pg
from pequegrad import Tensor, device, dt, grads
import numpy as np
import torch
from torch import tensor as torch_tensor, Tensor as TorchTensor
import pytest

dtypemapnp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}
dtypemapt = {
    dt.float32: torch.float32,
    dt.float64: torch.float64,
    dt.int32: torch.int32,
}


def all_devices(func):
    return pytest.mark.parametrize("device", ["cpu", "cuda"])(func)


def _compare_fn_with_torch(
    shapes,
    pequegrad_fn,
    torch_fn=None,
    tol: float = 1e-5,
    backward=True,
    device="cpu",
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
        nparr = np.random.uniform(low=0.5, high=0.9, size=torch_res.shape)
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


class TestNew:
    def test_fill(self):
        a = pg.fill((2, 3), dt.float32, 1, device.cpu(0))
        assert np.allclose(a.to_numpy(), np.ones((2, 3)))

    def test_custom(self):
        def torch_fn(a, b, c, d):
            return (a * b + c * d) * d

        def pq_fn(a, b, c, d):
            return pg.mul(pg.add(pg.mul(a, b), pg.mul(c, d)), d)

        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)

    def test_custom2(self):
        def torch_fn(a, b, c):
            return torch.mul(torch.add(torch.mul(a, b), c), b)

        def pq_fn(a, b, c):
            return pg.mul(pg.add(pg.mul(a, b), c), b)

        _compare_fn_with_torch([(2, 3), (2, 3), (2, 3)], pq_fn, torch_fn)

    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize(
        "dtype", [dt.float32, dt.float64]
    )  # TODO -- correct Int32 implementations (for example, divs need to be casted to float before division)
    @all_devices
    @pytest.mark.parametrize(
        "lambdaop",
        [
            (lambda x, y: pg.add(x, y), lambda x, y: torch.add(x, y), True, False),
            (lambda x, y: pg.mul(x, y), lambda x, y: torch.mul(x, y), True, False),
            (lambda x, y: pg.sub(x, y), lambda x, y: torch.sub(x, y), True, False),
            (lambda x, y: pg.div(x, y), lambda x, y: torch.div(x, y), True, False),
            (lambda x, y: pg.pow(x, y), lambda x, y: torch.pow(x, y), True, False),
            (lambda x, y: pg.gt(x, y), lambda x, y: torch.gt(x, y), False, False),
            (lambda x, y: pg.lt(x, y), lambda x, y: torch.lt(x, y), False, False),
            (lambda x, y: pg.neq(x, y), lambda x, y: torch.ne(x, y), False, False),
            (lambda x, y: pg.max(x, y), lambda x, y: torch.max(x, y), True, False),
        ],
    )
    def test_binary_ops(self, shape, dtype, lambdaop, device):
        pq_fn, torch_fn, do_backward_float, do_backward_on_int = lambdaop
        _compare_fn_with_torch(
            [shape, shape],
            pq_fn,
            torch_fn,
            backward=do_backward_float
            if dtype in [dt.float32, dt.float64]
            else do_backward_on_int
            if dtype == dt.int32
            else False,
            device=device,
            dtype=dtype,
        )

    # unary ops
    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64, dt.int32])
    @all_devices
    @pytest.mark.parametrize(
        "lambdaop",
        [
            (lambda x: pg.log(x), lambda x: torch.log(x), True),
            (lambda x: pg.exp(x), lambda x: torch.exp(x), True),
            (lambda x: pg.sin(x), lambda x: torch.sin(x), True),
            (lambda x: pg.cos(x), lambda x: torch.cos(x), True),
        ],
    )
    def test_unary_ops(self, shape, dtype, lambdaop, device):
        pq_fn, torch_fn, do_backward = lambdaop
        _compare_fn_with_torch(
            [shape], pq_fn, torch_fn, backward=do_backward, device=device
        )

    # REDUCERS TESTS
    @all_devices
    @pytest.mark.parametrize("shape", [(2, 3), (3, 4), (4, 5)])
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @pytest.mark.parametrize("axes", [(0, 1), (1, 0), (0,), (1,), (-1), (-1, -2), None])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize(
        "lambdaop",
        [
            (
                lambda x, axes, keepdims: pg.sum(x, axes, keepdims),
                lambda x, axes, keepdims: torch.sum(x, dim=axes, keepdim=keepdims),
                True,
            ),
            (
                lambda x, axes, keepdims: pg.mean(x, axes, keepdims),
                lambda x, axes, keepdims: torch.mean(x, dim=axes, keepdim=keepdims),
                True,
            ),
            (
                lambda x, axes, keepdims: pg.max_reduce(x, axes, keepdims),
                lambda x, axes, keepdims: torch.max(x, dim=axes, keepdim=keepdims)[0],
                True,
            ),
        ],
    )
    def test_reducers(self, device, shape, dtype, axes, keepdims, lambdaop):
        def pq_fn(x):
            return pg.sum(x, axes, keepdims)

        def torch_fn(x):
            return torch.sum(x, dim=axes, keepdim=keepdims)

        _compare_fn_with_torch(
            [shape], pq_fn, torch_fn, backward=lambdaop[2], device=device
        )

    # Test broadcast to
    @pytest.mark.parametrize(
        "shapes",
        [
            ((2, 3), (4, 2, 3)),
            ((2, 3), (2, 3)),
            ((2, 3), (1, 2, 3)),
            ((1, 2, 3), (2, 2, 3)),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    def test_broadcast_to(self, shapes, dtype):
        shape, target_shape = shapes

        def pq_fn(x):
            return pg.broadcast_to(x, target_shape)

        def torch_fn(x):
            return torch.broadcast_to(x, target_shape)

        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=False)

    # Test permute
    @pytest.mark.parametrize(
        "shape_and_dims",
        [
            ((2, 3), (1, 0)),
            ((2, 3, 4), (2, 0, 1)),
            ((2, 3, 4), (0, 1, 2)),
            ((4, 8, 15, 12, 3), (4, 0, 2, 1, 3)),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.int32, dt.float32, dt.float64])
    def test_permute(self, shape_and_dims, dtype):
        shape, dims = shape_and_dims

        def pq_fn(x):
            return pg.permute(x, dims)

        def torch_fn(x):
            return torch.permute(x, dims)

        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True)

    @pytest.mark.parametrize(
        "data",
        [
            [(1, 2, 3), 0, 1],
        ],
    )
    @all_devices
    def test_transpose(self, data, device):
        shape, dim0, dim1 = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.transpose(dim0, dim1),
            lambda x: x.transpose(dim0, dim1),
            device=device,
        )

    # Test matmul
    @pytest.mark.parametrize(
        "shapes",
        [
            ((4,), (4,)),
            ((4, 3), (3, 4)),
            ((5, 2, 10), (5, 10, 6)),
        ],
    )
    @all_devices
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    def test_matmul(self, shapes, dtype, device):
        def pq_fn(a, b):
            return pg.matmul(a, b)

        def torch_fn(a, b):
            return torch.matmul(a, b)

        _compare_fn_with_torch(shapes, pq_fn, torch_fn, backward=True, device=device)

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @all_devices
    @pytest.mark.parametrize(
        "lambdaop",
        [
            (
                lambda cond, x, y: pg.where(cond, x, y),
                lambda cond, x, y: torch.where(cond.bool(), x, y),
                False,
            ),  # backward not implemented
        ],
    )
    def test_where(self, shape, dtype, lambdaop, device):
        pq_fn, torch_fn, do_backward = lambdaop
        _compare_fn_with_torch(
            [shape, shape, shape], pq_fn, torch_fn, backward=do_backward, device=device
        )

    @pytest.mark.parametrize(
        "shape_and_dims",
        [
            ((2, 1, 3), 1),
            ((2, 1, 1), (1, 2)),
            ((2, 1, 1), 1),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.int32, dt.float32, dt.float64])
    def test_squeeze(self, shape_and_dims, dtype):
        shape, dims = shape_and_dims

        def pq_fn(x):
            return pg.squeeze(x, dims)

        def torch_fn(x):
            return torch.squeeze(x, dims)

        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True)

    @pytest.mark.parametrize(
        "shape_and_dims",
        [
            ((2, 3), 1),
            ((2, 3), 0),
            ((2, 3), -1),
            ((2, 3), -2),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.int32, dt.float32, dt.float64])
    def test_unsqueeze(self, shape_and_dims, dtype):
        shape, dims = shape_and_dims

        def pq_fn(x):
            return pg.unsqueeze(x, dims)

        def torch_fn(x):
            return torch.unsqueeze(x, dims)

        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True)

    # test im2col
    @pytest.mark.parametrize(
        "info",  # shape, kernel_size, stride, dilation
        [
            ((2, 3, 20, 20), (3, 3), (1, 1), (1, 1)),
            ((2, 3, 20, 20), (3, 3), (2, 2), (1, 1)),
            ((2, 3, 20, 20), (3, 3), (1, 1), (2, 2)),
            ((2, 3, 20, 20), (3, 3), (2, 2), (2, 2)),
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64])
    @all_devices
    def test_im2col(self, info, dtype, device):
        shape, kernel_size, stride, dilation = info

        im2col_out_shape = None

        def pq_fn(x):
            return pg.im2col(
                x, kernel_size, stride, (), dilation
            )  # padding is not implemented yet

        def torch_fn(x):
            nonlocal im2col_out_shape
            a = torch.nn.functional.unfold(
                x, kernel_size, dilation=dilation, stride=stride, padding=0
            )
            im2col_out_shape = a.shape
            return a

        _compare_fn_with_torch([shape], pq_fn, torch_fn, backward=True, device=device)

        # Now col2im
        def pq_fn(x):
            return pg.col2im(x, shape[2:], kernel_size, stride, (), dilation)

        def torch_fn(x):
            return torch.nn.functional.fold(
                x, shape[2:], kernel_size, dilation=dilation, stride=stride, padding=0
            )

        _compare_fn_with_torch(
            [im2col_out_shape], pq_fn, torch_fn, backward=True, device=device
        )

        # test slice + slice autograd

    @pytest.mark.parametrize(
        "data",  # tensor_shape, slices(arr of slices)
        [
            [(3, 3), (slice(0, 2), slice(0, 2))],
            # stepped slices
            [(9, 11), (slice(0, 9, 2), slice(0, 11, 3))],
            # slice with array
            [(3, 3), (slice(0, 2), [0, 1])],
            # mix
            [(3, 10, 5), (slice(0, 2), slice(0, 10), [0, 1])],
            [(7, 5, 8), (1, slice(0, 5), slice(0, 8))],
            # slice len < tensor ndim
            [(3, 3), (slice(0, 2))],
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64, dt.int32])
    @all_devices
    def test_select(self, data, dtype, device):
        tensor_shape, slices = data

        def torch_fn(x):
            return x[slices]

        def peq_fn(x):
            return x[slices]

        _compare_fn_with_torch(
            [tensor_shape],
            peq_fn,
            torch_fn,
            backward=True,
            device=device,
        )

    @pytest.mark.parametrize(
        "data",  # tensor_shape, slices(arr of slices), "setitem" shape
        [
            [(3, 3), (slice(0, 2), slice(0, 2)), (2, 2)],
            # stepped slices
            [(9, 11), (slice(0, 9, 2), slice(0, 11, 3)), (5, 4)],
            # slice with array
            [(3, 3), (slice(0, 2), [0, 1]), (2, 2)],
            # mix
            [(3, 10, 5), (slice(0, 2), slice(0, 10), [0, 1]), (2, 10, 2)],
            [(7, 5, 8), (1, slice(0, 5), slice(0, 8)), (5, 8)],
            # slice len < tensor ndim
            [(3, 3), (slice(0, 2)), (2, 3)],
        ],
    )
    @pytest.mark.parametrize("dtype", [dt.float32, dt.float64, dt.int32])
    @all_devices
    def test_assign_at(self, data, dtype, device):
        tensor_shape, slices, assign_at_shape = data

        def torch_fn(x, y):
            x = x.clone()
            x[slices] = y
            return x

        def peq_fn(x, y):
            z = pg.assign_at(x, y, slices)
            return z

        _compare_fn_with_torch(
            [tensor_shape, assign_at_shape],
            peq_fn,
            torch_fn,
            backward=False,
            device=device,
        )
