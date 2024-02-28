from pequegrad.storage import AbstractStorage, NumpyStorage, CudaStorage
from pequegrad.cuda import CUDA_AVAILABLE
import pytest
import torch
import numpy as np

# TODO -- Add tests for non-contiguous tensors


storages_to_test = [NumpyStorage]
if CUDA_AVAILABLE:
    storages_to_test.append(CudaStorage)

np.random.seed(42)


class _TestStorage:
    dtype: any

    def _compare_with_numpy(
        self, x: AbstractStorage, y: np.ndarray, test_strides=True, tol=1e-5
    ):
        assert x.shape == y.shape
        if test_strides:
            assert x.strides == y.strides
        print(x.numpy(), "\n<========================>\n", y)
        # assert x.dtype == y.dtype
        np.testing.assert_allclose(x.numpy(), y, rtol=tol)

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1), tuple()]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize(
        "lambdaop",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: x == y,
            lambda x, y: x != y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x > y,
            lambda x, y: x >= y,
            lambda x, y: x**y,
            [lambda x, y: x.el_wise_max(y), np.maximum],
        ],
    )
    def test_binop(self, shape, class_storage, lambdaop):
        lambdaopnp = lambdaop[1] if isinstance(lambdaop, list) else lambdaop
        lambdaoptensor = lambdaop[0] if isinstance(lambdaop, list) else lambdaop

        # random.randn returns a Python float, we need to wrap it in np.array
        np1 = np.random.rand(*shape) * 2 + 1  # to avoid division by zero
        np1 = np.array(np1, dtype=self.dtype)
        np2 = np.random.rand(*shape) * 2 + 1  # to avoid division by zero
        np2 = np.array(np2, dtype=self.dtype)

        x = class_storage(np1)
        y = class_storage(np2)
        res = lambdaoptensor(
            x, y
        )  # cast to float as of now (only in case of NPStorage)
        if class_storage == NumpyStorage:
            assert (
                type(res) == NumpyStorage
            ), "Result should be of type NPStorage, op: " + str(lambdaop)
            res.data = res.data.astype(self.dtype)
        self._compare_with_numpy(res, lambdaopnp(np1, np2).astype(self.dtype))

    # test for broadcasting shapes -> from, to
    @pytest.mark.parametrize(
        "shape",
        [
            [(3, 4), (1, 3, 4)],
            [(5,), (3, 5)],
            [(1, 2, 3), (4, 1, 2, 3)],
            [(3, 1), (3, 4)],
            [(1,), (7, 1)],
            [(1, 3, 1), (2, 1, 3, 4)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_broadcast_to(self, shape, class_storage):
        from_shape, to_shape = shape
        nparr = np.random.rand(*from_shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(
            x.broadcast_to(to_shape), np.broadcast_to(nparr, to_shape)
        )

    # test for binary operations with broadcasting
    @pytest.mark.parametrize(
        "shape",
        [
            [(3, 4), (1, 3, 4)],
            [(5,), (3, 5)],
            [(1, 2, 3), (4, 1, 2, 3)],
            [(3, 1), (3, 4)],
            [(1,), (7, 1)],
            [(1, 3, 1), (2, 1, 3, 4)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize(
        "lambdaop",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x, y: x == y,
            lambda x, y: x != y,
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x > y,
            lambda x, y: x >= y,
            lambda x, y: x**y,
            [lambda x, y: x.el_wise_max(y), np.maximum],
        ],
    )
    def test_binop_broadcast(self, shape, class_storage, lambdaop):
        lambdaopnp = lambdaop[1] if isinstance(lambdaop, list) else lambdaop
        lambdaoptensor = lambdaop[0] if isinstance(lambdaop, list) else lambdaop

        from_shape, to_shape = shape
        nparr = (
            np.random.rand(*from_shape).astype(self.dtype) * 2 + 1
        )  # to avoid division by zero
        nparrbroadcasted = (
            np.random.rand(*to_shape).astype(self.dtype) * 2 + 1
        )  # to avoid division by zero
        x = class_storage(nparr)
        y = class_storage(nparrbroadcasted)
        res = lambdaoptensor(
            x, y
        )  # cast to float as of now (only in case of NPStorage)
        if class_storage == NumpyStorage:
            assert (
                type(res) == NumpyStorage
            ), "Result should be of type NumpyStorage, got: " + str(type(res))
            res.data = res.data.astype(self.dtype)
        self._compare_with_numpy(
            res, lambdaopnp(nparr, nparrbroadcasted).astype(self.dtype)
        )

    @pytest.mark.parametrize(
        # shape, new_order
        "data",
        [[(3, 4), (1, 0)], [(5,), (0,)], [(1, 2, 3), (2, 0, 1)]],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_transpose_and_contiguous(self, data, class_storage):
        shape, new_order = data
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        x_permuted = x.permute(*new_order)
        np_transposed = np.transpose(nparr, new_order)
        self._compare_with_numpy(x_permuted, np_transposed)

        # test for contiguous
        npcont = np.array(
            np_transposed.data, order="C"
        )  # this will copy the array in a way that it is contiguous
        xcont = x_permuted.contiguous()
        self._compare_with_numpy(xcont, npcont)

        assert xcont.is_contiguous()
        if len(shape) > 1:
            assert not x_permuted.is_contiguous()
            assert xcont.is_contiguous()
        else:
            assert x_permuted.is_contiguous()
            assert xcont.is_contiguous()

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize(
        "lambdaop",  # tensor, np
        [
            [lambda x: x.log(), np.log],
            [lambda x: x.exp(), np.exp],
        ],
    )
    def test_unary_op(self, shape, class_storage, lambdaop):
        if self.dtype == np.int32:
            raise pytest.skip("Unary ops not tested properly for int32")
        tensor_op, np_op = lambdaop

        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        res = tensor_op(x)
        self._compare_with_numpy(res, np_op(nparr))

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_T(self, shape, class_storage):
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(x.T, np.transpose(nparr, axes=range(len(shape))[::-1]))

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_ndim(self, shape, class_storage):
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        assert x.ndim == len(shape)

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_size(self, shape, class_storage):
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        assert x.size == np.prod(shape)

    @pytest.mark.parametrize(
        "shape_and_axistoswap",
        [
            [(3, 4), (1, 0)],
            [(1, 2, 3), (2, 0, 1)],
            [(1, 2, 3), (0, 2)],
            [(3, 1), (0, 1)],
            [(1, 3, 1), (0, 2)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_swapaxes(self, shape_and_axistoswap, class_storage):
        shape, axis_to_swap = shape_and_axistoswap
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(
            x.swapaxes(axis1=axis_to_swap[0], axis2=axis_to_swap[1]),
            np.swapaxes(nparr, axis_to_swap[0], axis_to_swap[1]),
        )

    # matmul tests
    @pytest.mark.parametrize(
        "shapes",
        [
            [(3, 3), (3, 3)],
            [(3, 4), (4, 5)],
            [(3, 4), (4,)],
            [(4,), (4, 3)],
            [(25, 20), (20, 25)],
            [(22,), (22, 30)],
            [(2, 3, 1), (2, 1, 3)],
            [(20, 5, 30), (20, 30, 5)],
            [(1, 5, 10), (10, 10, 5)],
            [(3,), (3,)],  # vector x vector dot
            [(200,), (200,)],
            [
                (513,),
                (513,),
            ],
            [(500000,), (500000,)],
            [(5, 2, 2, 3), (5, 2, 3, 4)],
            [(2, 2, 3, 5), (5, 2)],
            [
                (2, 2, 3, 5),
                (
                    1,
                    1,
                    5,
                    2,
                ),
            ],
            [(8, 5, 30, 20), (8, 5, 20, 30)],
            [(4, 2, 3, 2, 5), (4, 2, 3, 5, 3)],
            [(1, 2, 3, 2, 5), (4, 2, 3, 5, 3)],
            [(5, 3), (2, 2, 3, 5)],
        ],
    )  # (from, to)
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_matmul(self, shapes, class_storage):
        from_shape, to_shape = shapes
        nparr = np.random.rand(*from_shape).astype(self.dtype)
        nparr2 = np.random.rand(*to_shape).astype(self.dtype)
        x = class_storage(nparr)
        y = class_storage(nparr2)
        self._compare_with_numpy(x.matmul(y), np.matmul(nparr, nparr2), tol=1e-3)

    # ternary operations
    @pytest.mark.parametrize(
        "shape",
        [
            (3, 4),
            (5,),
            (1, 2, 3),
            (3, 1),
            (1,),
            (1, 3, 1),
            (3, 4, 5),
            (1, 1, 1),
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize(
        "lambdaop",
        [
            [lambda x, y, z: x.where(y, z), lambda x, y, z: np.where(y, x, z)],
        ],
    )
    def test_ternary_op(self, shape, class_storage, lambdaop):
        lambdaopnp = lambdaop[1] if isinstance(lambdaop, list) else lambdaop
        lambdaoptensor = lambdaop[0] if isinstance(lambdaop, list) else lambdaop
        np1 = np.random.rand(*shape).astype(self.dtype)
        np2 = np.random.rand(*shape).astype(self.dtype)
        np3 = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(np1)
        y = class_storage(np2)
        z = class_storage(np3)
        res = lambdaoptensor(x, y, z)
        self._compare_with_numpy(res, lambdaopnp(np1, np2, np3))

    @pytest.mark.parametrize(
        "params",  # shape, dim, keepdims
        [
            # test for single dimension
            [(3, 4), 0, True],
            [(3, 4), 0, False],
            [(3, 4), 1, True],
            [(3, 4), 1, False],
            [(3, 4, 2), 2, True],
            [(3, 4, 2), 2, False],
            [(3, 4, 2), 1, True],
            [(3, 4, 2), 1, False],
            [(3, 4, 2), 0, True],
            [(3, 4, 2), 0, False],
            [(3, 4, 2, 5), 3, True],
            [(3, 4, 2, 5), 3, False],
            [(3, 4, 2, 5), 2, True],
            [(3, 4, 2, 5), 2, False],
            [(3, 4, 2, 5), 1, True],
            [(3, 4, 2, 5), 1, False],
            [(3, 4, 2, 5), 0, True],
            [(3, 4, 2, 5), 0, False],
            [(5,), 0, True],
            [(5,), 0, False],
            [(2,), 0, True],
            [(2,), 0, False],
            [(2, 3, 5, 8, 9), 2, True],
            [(2, 3, 5, 8, 9), 2, False],
            [(2, 3, 5, 8, 9), 3, True],
            [(2, 3, 5, 8, 9), 3, False],
            [(2, 3, 5, 8, 9), 4, True],
            [(2, 3, 5, 8, 9), 4, False],
            [(2, 3, 5, 8, 9), 1, True],
            [(2, 3, 5, 8, 9), 1, False],
            [(2, 3, 5, 8, 9), 0, True],
            [(2, 3, 5, 8, 9), 0, False],
            # test for 'all dimensions'
            [(3, 4), None, True],
            [(3, 4), None, False],
            [(3, 4, 2), None, True],
            [(3, 4, 2), None, False],
            [(3, 4, 2, 5), None, True],
            [(3, 4, 2, 5), None, False],
            [(5,), None, True],
            [(5,), None, False],
            [(2,), None, True],
            [(2,), None, False],
            [(2, 3, 5, 8, 9), None, True],
            [(2, 3, 5, 8, 9), None, False],
            # test for tuples of dimensions
            [(3, 4), (0, 1), True],
            [(3, 4), (0, 1), False],
            [(3, 4, 2), (0, 2), True],
            [(3, 4, 2), (0, 2), False],
            [(3, 4, 2, 5), (1, 3), True],
            [(3, 4, 2, 5), (1, 3), False],
            [(5,), (0,), True],
            [(5,), (0,), False],
            [(2,), (0,), True],
            [(2,), (0,), False],
            [(2, 3, 5, 8, 9), (2, 3), True],
            [(2, 3, 5, 8, 9), (2, 3), False],
            [(2, 3, 5, 8, 9), (0, 1, 2, 3, 4), True],
            [(2, 3, 5, 8, 9), (0, 1, 2, 3, 4), False],
            [(1, 2, 3, 4, 5), 3, False],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize("op", ["sum", "max", "mean"])
    def test_reduce(self, params, class_storage, op):
        shape, axis, keepdims = params

        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)

        if op == "sum":
            self._compare_with_numpy(
                x.sum(axis=axis, keepdims=keepdims),
                np.sum(nparr, axis=axis, keepdims=keepdims, dtype=self.dtype),
            )
        elif op == "max":
            self._compare_with_numpy(
                x.max(axis=axis, keepdims=keepdims),
                np.max(nparr, axis=axis, keepdims=keepdims).astype(self.dtype),
            )
        elif op == "mean":
            self._compare_with_numpy(
                x.mean(axis=axis, keepdims=keepdims),
                np.mean(nparr, axis=axis, keepdims=keepdims).astype(self.dtype),
            )

    # test squeeze
    @pytest.mark.parametrize(
        "shape", [[(3, 1, 4), 1], [(1, 2, 3), 0], [(3, 2, 1), -1], [(1, 2, 3), -3]]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_squeeze(self, shape, class_storage):
        shape, axis = shape
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(x.squeeze(axis), np.squeeze(nparr, axis))

    @pytest.mark.parametrize(
        "shape",
        [
            [(3, 1, 4), 0],
            [(1, 2, 3), 1],
            [(1, 2, 3), 2],
            [(1, 2, 3), (0, 2)],
            [(1, 2, 3), (-1, 2)],
            [(1, 2, 3), (-1, 0)],
            [(1, 2, 3), (-1, -2)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_squeeze_invalid(self, shape, class_storage):
        shape, axis = shape
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        with pytest.raises(BaseException):
            x.squeeze(axis)

    # test unsqueeze
    @pytest.mark.parametrize(
        "shape",
        [
            [(3, 4, 3), 1],
            [(1, 2, 3), 0],
            [(1, 2, 3), 2],
            [(1, 2, 3), (0, 2)],
            [(1, 2, 3), (-1, 2)],
            [(1, 2, 3), (-1, 0)],
            [(1, 2, 3), (-1, -2)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_expand_dims(self, shape, class_storage):
        shape, axis = shape
        nparr = np.random.rand(*shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(x.expand_dims(axis), np.expand_dims(nparr, axis))

    # test reshape
    @pytest.mark.parametrize(
        "shape",
        [
            [(3, 4), (2, 6)],
            [(3, 4), (12,)],
            [(3, 4), (3, 4)],
            [(3, 4), (4, 3)],
            [(3, 4), (-1, 12)],
            [(3, 4), (3, -1)],
            [(3, 4), (-1, 4)],
            [(2, 4, 8), (2, -1, 4)],
            [(2, 4, 8), (2, 4, -1)],
            [(2, 4, 8), (-1, 2)],
            [(2, 4, 8), (2, -1, 8)],
            [(2, 4, 8), (-1, 4, 8)],
            [(2, 4, 8), (-1, 2, 4, 8)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_reshape(self, shape, class_storage):
        from_shape, to_shape = shape
        nparr = np.random.rand(*from_shape).astype(self.dtype)
        x = class_storage(nparr)
        self._compare_with_numpy(x.reshape(*to_shape), np.reshape(nparr, to_shape))

    # outer product(vector x vector)
    @pytest.mark.parametrize(
        "shape",
        [
            [(3,), (4,)],
            [(5,), (3,)],
            [(4,), (4,)],
            [(1000,), (1000,)],
            [(2000,), (3000,)],
            [(3000,), (2000,)],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_outer_product(self, shape, class_storage):
        nparr = np.random.rand(*shape[0]).astype(self.dtype)
        nparr2 = np.random.rand(*shape[1]).astype(self.dtype)
        x = class_storage(nparr)
        y = class_storage(nparr2)
        self._compare_with_numpy(x.outer_product(y), np.outer(nparr, nparr2))

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
            [(1, 1, 3, 3), (3, 3), 1],
            [(1, 1, 5, 5), (3, 3), 1],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_im2col(self, data, class_storage):
        if self.dtype != np.float32:
            raise pytest.skip("im2col only supported for float32")
        shape_input, kernel_size, stride = data

        nparr = np.random.rand(*shape_input).astype(self.dtype)
        x = class_storage(nparr)
        x_transformed = x.im2col(kernel_size, stride)
        x_torch = torch.tensor(nparr)
        x_torch_transformed = torch.nn.functional.unfold(
            x_torch, kernel_size, stride=stride
        )
        self._compare_with_numpy(
            x_transformed,
            x_torch_transformed.numpy().astype(self.dtype),
            test_strides=False,
        )

    @pytest.mark.parametrize(
        "data",
        # shape_input, kernel_size
        [
            [(2, 2, 3, 3), (2, 2), 1],
            [(2, 2, 3, 3), (2, 2), 2],
            [(1, 2, 10, 5), (3, 3), 1],
            [(1, 4, 10, 5), (1, 1), 1],
            [(5, 2, 10, 5), (3, 3), 1],
            [(5, 1, 10, 5), (3, 3), 2],
            [(5, 1, 10, 5), (1, 1), 1],
            [(5, 6, 10, 5), (5, 5), 1],
            [(1, 1, 3, 3), (3, 3), 1],
        ],
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    def test_col2im(self, data, class_storage):
        if self.dtype != np.float32:
            raise pytest.skip("col2im only supported for float32")
        shape_input, kernel_size, stride = data

        nparr = np.random.rand(*shape_input).astype(self.dtype)
        torch_unfolded = torch.nn.functional.unfold(
            torch.tensor(nparr), kernel_size, stride=stride
        )
        torch_folded = (
            torch.nn.functional.fold(
                torch_unfolded, shape_input[2:], kernel_size, stride=stride
            )
            .detach()
            .numpy()
        )

        unfolded = class_storage(torch_unfolded.detach().numpy())
        folded = unfolded.col2im(kernel_size, shape_input[2:], stride)
        self._compare_with_numpy(folded, torch_folded, test_strides=False)

    # test fill
    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", storages_to_test)
    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0, 3.0, 4.0])
    def test_fill(self, shape, class_storage, value):
        value = self.dtype(value) # todo -- handle casts?
        x = class_storage.fill(shape, value, dtype=self.dtype)
        self._compare_with_numpy(x, np.full(shape, value, dtype=self.dtype))


class TestStorageFloat32(_TestStorage):
    dtype = np.float32


class TestStorageInt32(_TestStorage):
    dtype = np.int32


class TestStorageFloat64(_TestStorage):
    dtype = np.float64
