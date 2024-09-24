from pequegrad import Tensor, device
import numpy as np


def test_to():
    x = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    assert np.allclose(x.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
    a = x.to("cpu")
    assert a.device == device.cpu(0)
    assert np.allclose(a.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
    a = x.to("cuda")
    assert a.device == device.cuda(0)
    assert np.allclose(a.to_numpy(), np.array([1, 2, 3], dtype=np.float32))
