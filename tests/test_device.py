from pequegrad.backend.c import Tensor, device
import numpy as np


def test_to():
    x = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(x.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
    a = x.to(device.cpu)
    assert a.device == device.cpu
    assert np.allclose(a.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
    a = x.to(device.cuda)
    assert a.device == device.cuda
    assert np.allclose(a.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
