from pequegrad.storage import Storage
from pequegrad.cuda_storage import Storage as CudaStorage
import pytest
import numpy as np

# TODO -- USE a general interface
NPStorage = Storage


class TestStorage:

    def _compare_with_numpy(self, x: Storage, y: np.ndarray):
        assert np.allclose(x.numpy(), y)

    @pytest.mark.parametrize("shape", [(1, 2), (3, 4)])
    @pytest.mark.parametrize("class_storage", [NPStorage, CudaStorage])
    def test_add(self, shape, class_storage):
        np1 = np.random.rand(*shape)
        np2 = np.random.rand(*shape)
        x = class_storage(np1)
        y = class_storage(np2)
        self._compare_with_numpy(x + y, np1 + np2)
