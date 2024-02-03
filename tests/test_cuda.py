from pequegrad.cuda_storage import Storage
import numpy as np


def test_cuda_params():
    x = np.random.rand(10, 5).astype(np.float32)  # our storage will cast to float32
    y = Storage(x)
    assert x.shape == y.shape
    assert x.strides == y.strides


def test_cuda_getitem():
    x = np.random.rand(10, 5)
    y = Storage(x)
    assert np.allclose(x[0, 0], y[0, 0])
