from pequegrad.storage import Storage
from pequegrad.cuda_storage import Storage as CudaStorage
import pytest
import numpy as np

# We use float32 because it is the only dtype supported by CudaArray
NPStorage = Storage


class TestStorage:
    def _compare_with_numpy(self, x: Storage, y: np.ndarray):
        assert x.shape == y.shape
        assert x.strides == y.strides
        assert np.allclose(x.numpy(), y)

    @pytest.mark.parametrize(
        "shape", [(3, 4), (5,), (1, 2, 3), (3, 1), (1,), (1, 3, 1)]
    )
    @pytest.mark.parametrize("class_storage", [NPStorage, CudaStorage])
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
        ],
    )
    def test_binop(self, shape, class_storage, lambdaop):
        np1 = np.random.rand(*shape).astype(np.float32)
        np2 = np.random.rand(*shape).astype(np.float32)
        x = class_storage(np1)
        y = class_storage(np2)
        res = lambdaop(x, y)  # cast to float as of now (only in case of NPStorage)
        if class_storage == NPStorage:
            assert (
                type(res) == NPStorage
            ), "Result should be of type NPStorage, op: " + str(lambdaop)
            res.data = res.data.astype(np.float32)
        self._compare_with_numpy(res, lambdaop(np1, np2).astype(np.float32))

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
    @pytest.mark.parametrize("class_storage", [NPStorage, CudaStorage])
    def test_broadcast_to(self, shape, class_storage):
        from_shape, to_shape = shape
        nparr = np.random.rand(*from_shape).astype(np.float32)
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
    @pytest.mark.parametrize("class_storage", [NPStorage, CudaStorage])
    @pytest.mark.parametrize(
        "lambdaop",
        [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
        ],
    )
    def test_binop_broadcast(self, shape, class_storage, lambdaop):
        from_shape, to_shape = shape
        nparr = np.random.rand(*from_shape).astype(np.float32)
        nparrbroadcasted = np.random.rand(*to_shape).astype(np.float32)
        x = class_storage(nparr)
        y = class_storage(nparrbroadcasted)
        self._compare_with_numpy(lambdaop(x, y), lambdaop(nparr, nparrbroadcasted))

    # test permute (transpose for numpy)
    @pytest.mark.parametrize(
        # shape, new_order
        "data",
        [[(3, 4), (1, 0)], [(5,), (0,)], [(1, 2, 3), (2, 0, 1)]],
    )
    @pytest.mark.parametrize("class_storage", [NPStorage, CudaStorage])
    def test_transpose_contiguous(self, data, class_storage):
        shape, new_order = data
        nparr = np.random.rand(*shape).astype(np.float32)
        x = class_storage(nparr)
        x_permuted = x.permute(*new_order)
        np_transposed = np.transpose(nparr, new_order)
        self._compare_with_numpy(x_permuted, np_transposed)
