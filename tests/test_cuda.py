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


def test_to_numpy():
    x = np.random.rand(10, 5)
    y = Storage(x)
    assert np.allclose(x, y.to_numpy())


def test_cuda_elwise_add():
    x = np.random.rand(10, 5)
    y = np.random.rand(10, 5)
    z = Storage(x) + Storage(y)
    znp = x + y

    assert np.allclose(z.to_numpy(), znp)
