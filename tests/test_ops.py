import pytest
from pequegrad.tensor import Tensor, CUDA_AVAILABLE
from torch import Tensor as TorchTensor, tensor as torch_tensor
import numpy as np
import torch


def _compare_fn_with_torch(
    shapes,
    pequegrad_fn,
    torch_fn=None,
    tol: float = 1e-5,
    backward=True,
    pq_backend: str = "np",
):
    # In cases where the api is the same, just use the same fn as pequegrad
    torch_fn = torch_fn or pequegrad_fn

    # Ensure deterministic results
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Use a uniform distribution to initialize the arrays with 'good numbers' so that there are no numerical stability issues
    np_arr = [np.random.uniform(low=0.5, high=0.9, size=shape) for shape in shapes]
    tensors = [
        Tensor(arr.astype(np.float64), requires_grad=True, backend=pq_backend)
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
        peq_res.backward(Tensor(nparr.astype(np.float64), backend=pq_backend))
        torch_res.backward(torch_tensor(nparr, dtype=torch.float64))

        for t, torch_t in zip(tensors, torch_tensors):
            _compare(t.grad, torch_t.grad, tol)


class _TestOps:
    backend: str

    @pytest.mark.parametrize(
        "shape", [(2, 3), (3, 2), (6, 1), (1, 6), (6,), (1, 1, 6), (1, 2, 3), (2, 3, 1)]
    )
    def test_reshape(self, shape):
        _compare_fn_with_torch(
            [shape],
            lambda x: x.reshape(shape),
            lambda x: x.reshape(shape),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "shapes",
        [
            [(3,), (3,)],
            [(2, 3), (2, 3)],
            [(1, 2, 3), (1, 2, 3)],
        ],
    )
    def test_pow(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x, y: x**y, lambda x, y: x**y, pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "data",
        [
            [(2, 3), 0],
            [(2, 3), 1],
            [(2, 3), None],
            [(2, 3), -1],
            [(2, 3), -2],
            [(1, 2, 3), None],
            [(1, 2, 3), -1],
            [(1, 2, 3), (-1, -2)],
        ],
    )
    def test_mean(self, data):
        shape, dim = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.mean(dim=dim, keepdim=False),
            lambda x: x.mean(dim=dim, keepdim=False),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "shapes",
        [
            [(1, 2, 3), (1, 2, 3)],
            [(1, 2, 3), (1, 2, 1)],
            [(1, 2, 3), (1, 1, 3)],
            [(2, 3), (3,)],
        ],
    )
    def test_add(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x, y: x + y, lambda x, y: x + y, pq_backend=self.backend
        )

    @pytest.mark.parametrize("shapes", [[(3,)], [(2, 3)], [(1, 2, 3)], [(2, 4, 1)]])
    def test_exp(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x: x.exp(), lambda x: x.exp(), pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "shape",
        [
            [(3,)],
            [
                (
                    3,
                    2,
                )
            ],
        ],
    )
    def test_mse(self, shape):
        def torch_fn(x, y):
            mse = torch.nn.MSELoss(reduction="mean")
            return mse(x, y)

        _compare_fn_with_torch(
            shape * 2, lambda x, y: x.mse_loss(y), torch_fn, pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "shapes",
        [
            [(1, 2, 3), (1, 2, 3)],
            [(1, 2, 3), (1, 2, 1)],
            [(1, 2, 3), (1, 1, 3)],
        ],
    )
    def test_mul(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x, y: x * y, lambda x, y: x * y, pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "shape",
        [
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
        ],
    )
    def test_relu(self, shape):
        _compare_fn_with_torch(shape, lambda x: x.relu(), pq_backend=self.backend)

    @pytest.mark.parametrize(
        "data",
        [
            [(2, 3), 0],
            [(2, 3), 1],
            [(2, 3), None],
            [(2, 3), -1],
            [(2, 3), -2],
            [(1, 2, 3), None],
            [(1, 2, 3), -1],
            [(1, 2, 3), (-1, -2)],
        ],
    )
    def test_sum(self, data):
        shape, dim = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.sum(
                dim=dim,
                keepdim=False,
            ),
            lambda x: x.sum(dim=dim, keepdim=False),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",
        [
            [(1, 2, 3), 0, 1],
        ],
    )
    def test_transpose(self, data):
        shape, dim0, dim1 = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.transpose(dim0, dim1),
            lambda x: x.transpose(dim0, dim1),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",  # shape, permutation_dims
        [
            [(1, 2, 3, 4), (0, 1, 2, 3)],
            [(1, 2, 3), (0, 1, 2)],
            [(1, 2, 3), (2, 1, 0)],
            [(1, 2, 3), (1, 2, 0)],
            [(1, 2, 3), (0, 2, 1)],
            [(1, 2, 3), (2, 0, 1)],
            [(1, 2, 3), (1, 0, 2)],
            [(1, 2), (0, 1)],
        ],
    )
    def test_permute(self, data):
        shape, permutation_dims = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.permute(*permutation_dims),
            lambda x: x.permute(*permutation_dims),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            [(3,)],
            [(2, 3)],
            [(1, 2, 3)],
        ],
    )
    def test_max(self, shape):
        _compare_fn_with_torch(
            shape, lambda x: x.max(), lambda x: x.max(), pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    def test_softmax(self, shape):
        _compare_fn_with_torch(
            shape,
            lambda x: x.softmax(dim=-1),
            lambda x: x.softmax(dim=-1),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    def test_log_softmax(self, shape):
        _compare_fn_with_torch(
            shape,
            lambda x: x.log_softmax(dim=-1),
            lambda x: x.log_softmax(dim=-1),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize("shape", [[(3,)], [(2, 3)], [(1, 2, 3)]])
    def test_cross_entropy_loss_probs(self, shape):
        def torch_fn(x, y):
            nn_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
            return nn_cross_entropy(x, y)

        _compare_fn_with_torch(
            shape * 2,
            lambda x, y: x.cross_entropy_loss_probs(y),
            torch_fn,
            pq_backend=self.backend,
        )

    # batch, classes
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (6, 1), (1, 6)])
    def test_cross_entropy_loss_index(self, shape):
        np_idx = np.random.randint(0, shape[1], size=shape[0]).astype(np.int64)
        correct_index = Tensor(np_idx)
        correct_index_torch = torch_tensor(np_idx)

        def torch_fn(x):
            nn_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
            return nn_cross_entropy(x, correct_index_torch)

        _compare_fn_with_torch(
            [shape],
            lambda x: x.cross_entropy_loss_indices(correct_index),
            torch_fn,
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",
        [
            [(3,), 0],
            [(2, 3), 1],
            [(1, 2, 3), 0],
            [(1, 2, 3), 1],
            [(1, 2, 3), 2],
        ],
    )
    def test_unsqueeze(self, data):
        shape, dim = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.unsqueeze(dim),
            lambda x: x.unsqueeze(dim),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",
        [
            [(1, 2, 3), 0],
            [(1, 3, 1), 2],
        ],
    )
    def test_squeeze(self, data):
        shape, dim = data
        _compare_fn_with_torch(
            [shape],
            lambda x: x.squeeze(dim),
            lambda x: x.squeeze(dim),
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "shapes",
        [
            [(3,)],
            [(2, 3)],
            [(1, 2, 3)],
            [(2, 4, 1)],
        ],
    )
    def test_log(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x: x.log(), lambda x: x.log(), pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "shapes",
        [
            [(3,), (3,)],
            [(2,), (2, 2)],
            [(2, 2), (2, 2)],
            [(4, 1), (1,)],
            [(4,), (4, 1)],
            [(2, 4, 1), (2, 1, 4)],  # both batched
            [(2, 4, 1), (1, 4)],  # batched x unbatched
            [(4, 1), (2, 1, 4)],  # unbatched x batched
            [(2, 2, 3, 5), (5, 2)],  # batched_2 x unbatched
            [(5, 2), (2, 2, 2, 5)],  # unbatched x batched_2
        ],
    )
    def test_matmul(self, shapes):
        _compare_fn_with_torch(
            shapes, lambda x, y: x @ y, lambda x, y: x @ y, pq_backend=self.backend
        )

    @pytest.mark.parametrize(
        "data",
        # shape_input, shape_kernel, bias, strides
        [
            # for input: batch_size, input_channels, input_height, input_width, dilation
            # for kernel: output_channels, input_channels, kernel_height, kernel_width, dilation
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 1, 1],
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 1, 2],
            [(1, 1, 10, 5), (1, 1, 3, 3), False, 2, 1],
            [(1, 1, 10, 5), (1, 1, 1, 1), True, 1, 1],
            [(1, 1, 10, 5), (1, 1, 1, 1), False, 2, 1],
            [(5, 1, 10, 5), (3, 1, 3, 3), True, 1, 1],
            [(5, 1, 10, 5), (3, 1, 3, 3), False, 2, 1],
            [(5, 1, 10, 5), (3, 1, 1, 1), True, 1, 1],
            [(5, 1, 10, 5), (3, 1, 1, 1), False, 2, 1],
            [(5, 1, 10, 5), (3, 1, 5, 5), True, 1, 1],
            [(5, 1, 10, 5), (3, 1, 5, 5), False, 2, 1],
            [(5, 3, 20, 10), (5, 3, 3, 3), True, 4, 1],
            [(5, 3, 20, 10), (5, 3, 3, 3), False, 50, 1],  # large stride
            # now with stride as tuple
            [(5, 3, 20, 10), (5, 3, 3, 3), True, (4, 2), 1],  # 12
            [(1, 3, 20, 10), (1, 3, 3, 3), True, (3, 3), 1],
            [(1, 3, 20, 10), (1, 3, 3, 3), False, (3, 1), 1],
            [(1, 3, 20, 10), (1, 3, 3, 3), True, (3, 3), 1],
            [(1, 3, 20, 10), (1, 3, 3, 3), False, (2, 5), 1],
        ],
    )
    def test_conv2d(self, data):
        shape_input, shape_kernel, use_bias, stride, dilation = data

        def torch_fn(x, y, b=None):
            if b is None:
                return torch.nn.functional.conv2d(
                    x, y, stride=stride, dilation=dilation
                )
            return torch.nn.functional.conv2d(
                x, y, bias=b, stride=stride, dilation=dilation
            )

        def peq_fn(x, y, b=None):
            if b is None:
                return x.conv2d(y, stride=stride, dilation=dilation)
            return x.conv2d(y, bias=b, stride=stride, dilation=dilation)

        if use_bias:
            bias_shape = (shape_kernel[0],)
        arr = [shape_input, shape_kernel]
        if use_bias:
            arr.append(bias_shape)
        _compare_fn_with_torch(
            arr,
            peq_fn,
            torch_fn,
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",
        # shape_input, kernel_size, stride
        [
            [(2, 2, 3, 3), (2, 2), 1],
            [(2, 2, 3, 3), (2, 2), 2],
            [(1, 1, 10, 5), (3, 3), 1],
            [(1, 1, 10, 5), (2, 2), 2],
            [(1, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (3, 3), 1],
            [(5, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (5, 5), 1],
        ],
    )
    def test_unfold(self, data):
        shape_input, kernel_size, stride = data

        def torch_fn(x):
            return torch.nn.functional.unfold(x, kernel_size, stride=stride)

        _compare_fn_with_torch(
            [shape_input],
            lambda x: x.unfold(kernel_size, stride=stride),
            torch_fn,
            pq_backend=self.backend,
        )

    @pytest.mark.parametrize(
        "data",
        # shape_input, kernel_size
        [
            [(2, 2, 3, 3), (2, 2), 1],
            [(2, 2, 3, 3), (2, 2), 2],
            [(1, 1, 10, 5), (3, 3), 1],
            [(1, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (3, 3), 1],
            [(5, 1, 10, 5), (3, 3), 2],
            [(5, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (5, 5), 1],
        ],
    )
    def test_fold(self, data):
        shape_input, kernel_size, stride = data

        def torch_fn(x):
            unfolded = torch.nn.functional.unfold(x, kernel_size, stride=stride)
            return torch.nn.functional.fold(
                unfolded, x.shape[2:], kernel_size, stride=stride
            )

        def peq_fn(x):
            unfolded = x.unfold(kernel_size, stride=stride)
            return unfolded.fold(kernel_size, x.shape[2:], stride=stride)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, pq_backend=self.backend)

    @pytest.mark.parametrize(
        "data",
        # shape_input, kernel_size, stride
        [
            [(1, 1, 10, 5), (3, 3), 1],
            [(1, 1, 10, 5), (3, 3), 2],
            [(1, 1, 10, 5), (1, 1), 4],
            [(1, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (3, 3), None],
            [(5, 1, 10, 5), (1, 1), 1],
            [(5, 1, 10, 5), (5, 5), None],
            [(5, 3, 10, 5), (5, 5), 5],
        ],
    )
    @pytest.mark.parametrize("method_name", ["avg_pool2d", "max_pool2d"])
    def test_pool2d(self, data, method_name):
        shape_input, kernel_size, stride = data

        def torch_fn(x):
            if stride is None:
                return torch.nn.functional.__getattribute__(method_name)(x, kernel_size)
            return torch.nn.functional.__getattribute__(method_name)(x, kernel_size, stride)

        def peq_fn(x):
            if stride is None:
                return x.__getattribute__(method_name)(kernel_size)
            return x.__getattribute__(method_name)(kernel_size, stride=stride)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, pq_backend=self.backend)

    @pytest.mark.parametrize(
        "data",
        # shape_input, dim, keepdim
        [
            [(5, 3, 10, 5), 0, True],
            [(5, 3, 10, 5), 1, True],
            [(5, 3, 10, 5), 2, True],
            [(5, 3, 10, 5), 3, True],
            [(5, 3, 10, 5), 0, False],
            [(5, 3, 10, 5), 1, False],
            [(5, 3, 10, 5), 2, False],
            [(5, 3, 10, 5), 3, False],
            [(5, 3, 10, 5), None, True],
            [(5, 3, 10, 5), None, False],
            [(5, 3, 10, 5), (0, 1), True],
            [(5, 3, 10, 5), (0, 1), False],
            [(5, 3, 10, 5), (1, 2), True],
            [(5, 3, 10, 5), (1, 2), False],
            [(5, 3, 10, 5), (2, 3), True],
            [(5, 3, 10, 5), (2, 3), False],
            [(5, 3, 10, 5), (0, 2), True],
            [(5, 3, 10, 5), (0, 2), False],
            [(5, 3, 10, 5), (1, 3), True],
            [(5, 3, 10, 5), (1, 3), False],
            [(5, 3, 10, 5), (0, 3), True],
            [(5, 3, 10, 5), (0, 3), False],
            [(5, 3, 10, 5), (0, 1, 2), True],
            [(5, 3, 10, 5), (0, 1, 2), False],
            [(5, 3, 10, 5), (1, 2, 3), True],
            [(5, 3, 10, 5), (1, 2, 3), False],
            [(5, 3, 10, 5), (0, 1, 3), True],
            [(5, 3, 10, 5), (0, 1, 3), False],
            [(5, 3, 10, 5), (0, 2, 3), True],
            [(5, 3, 10, 5), (0, 2, 3), False],
            [(5, 3, 10, 5), (0, 1, 2, 3), True],
            [(5, 3, 10, 5), (0, 1, 2, 3), False],
        ],
    )
    def test_std_var(self, data):
        shape_input, dim, keepdim = data

        def torch_fn(x):
            return torch.std(x, dim=dim, keepdim=keepdim)

        def peq_fn(x):
            return x.std(dim=dim, keepdim=keepdim)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, pq_backend=self.backend)

    @pytest.mark.parametrize(
        "data",
        # shape_input, normalized_shape, eps
        [
            [(5, 3, 10, 5), (3, 10, 5), 1e-5],
            [(2, 3, 4, 5), (3, 4, 5), 1e-5],
            [(2, 3, 4, 5), (4, 5), 1e-5],
            [(1, 2, 3, 4, 5), (2, 3, 4, 5), 1e-5],
            [(1, 2), (2,), 1e-5],
        ],
    )
    def test_layer_norm(self, data):
        shape_input, normalized_shape, eps = data

        def torch_fn(x):
            return torch.nn.functional.layer_norm(x, normalized_shape, eps=eps)

        def peq_fn(x):
            return x.layer_norm(normalized_shape, eps=eps)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, pq_backend=self.backend)

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
    def test_slice(self, data):
        tensor_shape, slices = data

        def torch_fn(x):
            return x[slices]

        def peq_fn(x):
            return x[slices]

        _compare_fn_with_torch(
            [tensor_shape],
            peq_fn,
            torch_fn,
            pq_backend=self.backend,
        )

    # test padding + padding autograd
    @pytest.mark.parametrize(
        "data",  # tensor_shape, padding, constant
        [
            [(3, 3), (1, 1, 1, 1), 0],  # padding all sides equally
            [(4, 4), (2, 2, 2, 2), 1],  # larger padding, constant 1
            [(5, 5), (0, 1, 2, 3), -1],  # asymmetric padding, negative constant
            [(6, 3), (0, 2, 0, 2), 0],  # padding left and right only
            [(7, 7), (1, 2, 3, 4), 2],  # different padding for each side, constant 2
        ],
    )
    # @pytest.mark.skip(reason="padding not implemented")
    def test_pad_constant(self, data):
        tensor_shape, padding, constant = data

        def torch_fn(x):
            return torch.nn.functional.pad(x, padding, value=constant)

        def peq_fn(x):
            return x.pad_constant(padding, constant)

        _compare_fn_with_torch(
            [tensor_shape],
            peq_fn,
            torch_fn,
            pq_backend=self.backend,
        )


# Run the tests for the different backend types
class TestOpsNP(_TestOps):
    backend = "np"


if CUDA_AVAILABLE:

    class TestOpsCuda(_TestOps):
        backend = "cuda"
