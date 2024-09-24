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


def test_get_available_devices():
    # hard coded
    assert device.get_available_devices() == [device.cpu(0), device.cuda(0)]


def test_force_emulated_devices():
    # hard coded
    assert device.get_available_devices() == [device.cpu(0), device.cuda(0)]

    # force
    device.force_emulated_devices(8, "cuda")

    assert device.get_available_devices() == [
        device.cpu(0),
        device.cuda(0),
        device.cuda(1),
        device.cuda(2),
        device.cuda(3),
        device.cuda(4),
        device.cuda(5),
        device.cuda(6),
        device.cuda(7),
    ]
