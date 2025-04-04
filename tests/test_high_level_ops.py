from pequegrad import Tensor, dt, grads
import pequegrad as pg
import numpy as np
import torch
from torch import tensor as torch_tensor, Tensor as TorchTensor
import pytest
from pequegrad import ops

dtypemapnp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}
dtypemapt = {
    dt.float32: torch.float32,
    dt.float64: torch.float64,
    dt.int32: torch.int32,
}


# decorator to parametrize devices
def all_devices(func):
    return pytest.mark.parametrize("device", ["cpu", "cuda"])(func)


def _compare_fn_with_torch(
    shapes,
    pequegrad_fn,
    torch_fn=None,
    tol: float = 1e-5,
    backward=True,
    device="cpu",
    dtype=dt.float32,
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


class TestNew:
    @pytest.fixture(autouse=True)
    def catch_exceptions(self, request):
        test_function = request.node.function
        # catch exception while running the test function
        try:
            # write to a file the name and args of the function
            """with open("test_results.txt", "a") as f:
            f.write(f"Running {test_function.__name__}\n")
            """
            yield
        except Exception as e:
            print(f"Exception in {test_function.__name__}")
            raise e

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
    @all_devices
    def test_layer_norm(self, data, device):
        shape_input, normalized_shape, eps = data

        def torch_fn(x):
            return torch.nn.functional.layer_norm(x, normalized_shape, eps=eps)

        def peq_fn(x):
            return pg.layer_norm(x, normalized_shape, eps=eps)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, tol=1e-4, device=device)

    @pytest.mark.parametrize(
        "data",
        # shape_input, size, alpha, beta, k
        [
            [(1, 1, 10, 5), 5, 1e-4, 0.75, 2],
            [(1, 1, 10, 5), 5, 1e-4, 0.75, 2],
            [(5, 1, 10, 5), 5, 1e-4, 0.75, 2],
            [(5, 3, 10, 5), 5, 1e-4, 0.75, 2],
            [(5, 3, 10, 5), 5, 1e-4, 0.75, 2],
        ],
    )
    @all_devices
    def test_local_response_norm(self, data, device):
        shape_input, size, alpha, beta, k = data

        def torch_fn(x):
            return torch.nn.functional.local_response_norm(x, size, alpha, beta, k)

        def peq_fn(x):
            return x.local_response_norm(size, alpha, beta, k)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, tol=1e-4, device=device)

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
    @all_devices
    def test_std_var(self, data, device):
        shape_input, dim, keepdim = data

        def torch_fn(x):
            return torch.std(x, dim=dim, keepdim=keepdim)

        def peq_fn(x):
            return x.std(dim=dim, keepdim=keepdim)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, device=device)

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
    @all_devices
    @pytest.mark.parametrize("method_name", ["avg_pool2d", "max_pool2d"])
    def test_pool2d(self, data, method_name, device):
        shape_input, kernel_size, stride = data

        def torch_fn(x):
            if stride is None:
                return torch.nn.functional.__getattribute__(method_name)(x, kernel_size)
            return torch.nn.functional.__getattribute__(method_name)(
                x, kernel_size, stride
            )

        def peq_fn(x):
            if stride is None:
                return x.__getattribute__(method_name)(kernel_size)
            return x.__getattribute__(method_name)(kernel_size, stride=stride)

        _compare_fn_with_torch([shape_input], peq_fn, torch_fn, device=device)

    @pytest.mark.parametrize(
        "data",
        # shape_input, shape_kernel, bias, strides, dilation, groups
        [
            # for input: batch_size, input_channels, input_height, input_width, dilation
            # for kernel: output_channels, input_channels, kernel_height, kernel_width, dilation
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 1, 1, 1],  # 0
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 1, 2, 1],
            [(1, 1, 10, 5), (1, 1, 3, 3), False, 2, 1, 1],
            [(1, 1, 10, 5), (1, 1, 1, 1), True, 1, 1, 1],
            [(1, 1, 10, 5), (1, 1, 1, 1), False, 2, 1, 1],
            [(5, 1, 10, 5), (3, 1, 3, 3), True, 1, 1, 1],  # 5
            [(5, 1, 10, 5), (3, 1, 3, 3), False, 2, 1, 1],
            [(5, 1, 10, 5), (3, 1, 1, 1), True, 1, 1, 1],
            [(5, 1, 10, 5), (3, 1, 1, 1), False, 2, 1, 1],
            [(5, 1, 10, 5), (3, 1, 5, 5), True, 1, 1, 1],  # 9
            [(5, 1, 10, 5), (3, 1, 5, 5), False, 2, 1, 1],
            [(5, 3, 20, 10), (5, 3, 3, 3), True, 4, 1, 1],
            [(5, 3, 20, 10), (5, 3, 3, 3), False, 50, 1, 1],  # large stride
            # now with stride as tuple
            [(5, 3, 20, 10), (5, 3, 3, 3), True, (4, 2), 1, 1],  # 12
            [(1, 3, 20, 10), (1, 3, 3, 3), True, (3, 3), 1, 1],  # 13
            [(1, 3, 20, 10), (1, 3, 3, 3), False, (3, 1), 1, 1],
            [(1, 3, 20, 10), (1, 3, 3, 3), True, (3, 3), 1, 1],
            [(1, 3, 20, 10), (1, 3, 3, 3), False, (2, 5), 1, 1],
            # 5 = 10 / groups
            [(2, 10, 20, 10), (8, 5, 3, 3), True, (1, 1), 1, 2],  # 17
        ],
    )
    @all_devices
    def test_conv2d(self, data, device):
        shape_input, shape_kernel, use_bias, stride, dilation, groups = data

        def torch_fn(x, y, b=None):
            if b is None:
                return torch.nn.functional.conv2d(
                    x, y, stride=stride, dilation=dilation, groups=groups
                )
            return torch.nn.functional.conv2d(
                x, y, bias=b, stride=stride, dilation=dilation, groups=groups
            )

        def peq_fn(x, y, b=None):
            if b is None:
                return x.conv2d(y, stride=stride, dilation=dilation, groups=groups)
            return x.conv2d(y, bias=b, stride=stride, dilation=dilation, groups=groups)

        if use_bias:
            bias_shape = (shape_kernel[0],)
        arr = [shape_input, shape_kernel]
        if use_bias:
            arr.append(bias_shape)
        _compare_fn_with_torch(arr, peq_fn, torch_fn, device=device)

    # test for transposed convolution
    @pytest.mark.parametrize(
        "data",
        # shape_input, shape_kernel, bias, strides, dilation, padding, output_padding
        [
            [(1, 1, 4, 4), (1, 1, 3, 3), False, 1, 1, 0, 0],
            [(1, 1, 4, 4), (1, 1, 3, 3), True, 1, 1, 0, 0],
            [(1, 1, 4, 4), (1, 1, 3, 3), False, 2, 1, 0, 0],
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 2, 2, 1, 0],
            [(1, 1, 10, 5), (1, 1, 3, 3), False, 2, 2, 2, 0],  # 4
            [(1, 1, 10, 5), (1, 1, 3, 3), True, 2, 2, 0, 1],
            [(1, 1, 3, 3), (1, 1, 3, 3), True, 2, 1, 0, 1],
        ],
    )
    @all_devices
    def test_conv2d_transpose(self, data, device):
        (
            shape_input,
            shape_kernel,
            use_bias,
            stride,
            dilation,
            padding,
            output_padding,
        ) = data

        def torch_fn(x, y, b=None):
            if b is None:
                return torch.nn.functional.conv_transpose2d(
                    x,
                    y,
                    stride=stride,
                    dilation=dilation,
                    output_padding=output_padding,
                    padding=padding,
                )
            return torch.nn.functional.conv_transpose2d(
                x,
                y,
                bias=b,
                stride=stride,
                dilation=dilation,
                output_padding=output_padding,
                padding=padding,
            )

        def peq_fn(x, y, b=None):
            if b is None:
                return x.conv_transpose2d(
                    y,
                    stride=stride,
                    dilation=dilation,
                    output_padding=output_padding,
                    padding=padding,
                )
            return x.conv_transpose2d(
                y,
                bias=b,
                stride=stride,
                dilation=dilation,
                output_padding=output_padding,
                padding=padding,
            )

        if use_bias:
            bias_shape = (shape_kernel[0],)
        arr = [shape_input, shape_kernel]
        if use_bias:
            arr.append(bias_shape)
        _compare_fn_with_torch(arr, peq_fn, torch_fn, device=device)

    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_softmax(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.softmax(dim=-1),
            lambda x: x.softmax(dim=-1),
            device=device,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_log_softmax(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.log_softmax(dim=-1),
            lambda x: x.log_softmax(dim=-1),
            device=device,
        )

    @pytest.mark.parametrize("shape", [[(3,)], [(2, 3)], [(1, 2, 3)]])
    @all_devices
    def test_cross_entropy_loss_probs(self, shape, device):
        def torch_fn(x, y):
            nn_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
            return nn_cross_entropy(x, y)

        _compare_fn_with_torch(
            shape * 2,
            lambda x, y: x.cross_entropy_loss_probs(y),
            torch_fn,
            device=device,
        )

    # batch, classes
    @pytest.mark.parametrize("shape", [(2, 3), (3, 2), (6, 1), (1, 6)])
    @all_devices
    def test_cross_entropy_loss_index(self, shape, device):
        np_idx = np.random.randint(0, shape[1], size=shape[0]).astype(np.int32)
        correct_index = Tensor(np_idx).to(device)
        correct_index_torch = torch_tensor(np_idx).long()

        def torch_fn(x):
            log_sm = torch.nn.functional.log_softmax(x, dim=-1)
            return -log_sm[:, correct_index_torch].mean()

        _compare_fn_with_torch(
            [shape],
            lambda x: x.cross_entropy_loss_indices(correct_index),
            torch_fn,
            device=device,
            backward=False,  # does not work yet :(
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
    @all_devices
    def test_pad_constant(self, data, device):
        tensor_shape, padding, constant = data

        def torch_fn(x):
            return torch.nn.functional.pad(x, padding, value=constant)

        def peq_fn(x):
            return x.pad_constant(padding, constant)

        _compare_fn_with_torch(
            [tensor_shape],
            peq_fn,
            torch_fn,
            device=device,
        )

    # tensordot tests
    @pytest.mark.parametrize(
        "data",
        # shape_input1, shape_input2, dim
        [
            [
                (2, 3, 4, 5),
                (8, 5, 4, 3),
                ([1, 3], [3, 1]),
            ],  # output shape: (2, 8, 3, 2)
            [
                (2, 3, 4, 5),
                (8, 5, 4, 3),
                ([1, 2], [3, 2]),
            ],  # output shape: (2, 5, 8, 5)
            [
                (100, 20),
                (20, 300),
                1,
            ],  # output shape: (100, 300), it is a matrix multiplication
            [(100, 20), (20, 300), 0],  # output shape: (100, 20, 20, 300), no reduction
        ],
    )
    @all_devices
    def test_tensordot(self, data, device):
        shape_input1, shape_input2, dim = data

        def torch_fn(x, y):
            return torch.tensordot(x, y, dims=dim)

        def peq_fn(x, y):
            return x.tensordot(y, dims=dim)

        _compare_fn_with_torch(
            [shape_input1, shape_input2],
            peq_fn,
            torch_fn,
            device=device,
        )

    # tests for simple einsum
    @pytest.mark.parametrize(
        "data",
        # equation, shape_input1, shape_input2
        [
            ["ij,jk->ik", (10, 20), (20, 30)],  # output shape: (10, 30)
            ["ij,jk->ik", (5, 5), (5, 5)],  # output shape: (5, 5)
            ["abc,bce->ae", (2, 3, 4), (3, 4, 6)],  # output shape: (2, 3, 6)
        ],
    )
    @all_devices
    def test_einsum(self, data, device):
        equation, shape_input1, shape_input2 = data

        def torch_fn(x, y):
            return torch.einsum(equation, x, y)

        def peq_fn(x, y):
            return pg.einsum(equation, x, y)

        _compare_fn_with_torch(
            [shape_input1, shape_input2],
            peq_fn,
            torch_fn,
            device=device,
        )

    @pytest.mark.skip(reason="Not implemented yet")
    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_erf(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.erf(),
            lambda x: x.erf(),
            device=device,
        )

    # tests for gelu
    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    @pytest.mark.parametrize("approximate", ["tanh"])
    def test_gelu(self, shape, device, approximate):
        _compare_fn_with_torch(
            shape,
            lambda x: x.gelu(approximate=approximate),
            lambda x: torch.nn.functional.gelu(x, approximate=approximate),
            device=device,
        )

    # tests for tanh
    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_tanh(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.tanh(),
            lambda x: torch.tanh(x),
            device=device,
        )

    # tests for sigmoid
    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_sigmoid(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.sigmoid(),
            lambda x: torch.sigmoid(x),
            device=device,
        )

    # tests for silu
    @pytest.mark.parametrize(
        "shape",
        [
            [(2, 3)],
        ],
    )
    @all_devices
    def test_silu(self, shape, device):
        _compare_fn_with_torch(
            shape,
            lambda x: x.silu(),
            lambda x: torch.nn.functional.silu(x),
            device=device,
        )

    # tests for cumsum
    @pytest.mark.parametrize(
        "info",
        [
            [(10,), 0],  # shape, dim
            [(10, 5), 0],
            [(10, 5), 1],
            [(10, 5, 3), 0],
            [(10, 5, 3), 1],
            [(10, 5, 3), 2],
            [(10, 5, 3), 0],
            [(10, 5, 3), 1],
            [(10, 5, 3), 2],
            [(10, 5, 3), 0],
            [(10, 5, 3), 1],
            [(10, 5, 3), 2],
        ],
    )
    @all_devices
    def test_cumsum(self, info, device):
        shape, dim = info
        _compare_fn_with_torch(
            [shape],
            lambda x: ops.cumsum(x, dim=dim),
            lambda x: torch.cumsum(x, dim=dim),
            device=device,
            backward=False,
        )

    # test scaled dot product attention
    @all_devices
    def test_scaled_dot_product_attention(self, device):
        # q, k, v, mask
        shape = [(5, 5, 10, 15), (5, 5, 10, 15), (5, 5, 10, 15)]
        m_ = pg.tril(Tensor.ones((5, 5, 10, 10)))

        def torch_fn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        def peq_fn(q, k, v):
            m = m_.to(q.device)
            return pg.scaled_dot_product_attention(q, k, v, m)

        _compare_fn_with_torch(shape, peq_fn, torch_fn, device=device, backward=False)
