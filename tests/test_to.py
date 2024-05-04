from pequegrad.tensor import Tensor, CUDA_AVAILABLE, device
import numpy as np
import pytest


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
class TestTo:
    @pytest.mark.parametrize("device", [device.cpu, device.cuda])
    def test_device(self, device):
        t = Tensor(np.array([1, 2, 3]), device=device)
        assert t.device == device

    def test_to(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, device=device.cpu)

        tcu = t.to(device.cuda)
        assert tcu.device == device.cuda
        assert tcu.numpy().all() == npdata.all()

        # test that to("cuda") does not modify the original tensor
        assert t.device == device.cpu
        assert t.numpy().all() == npdata.all()

        tnp = tcu.to(device.cpu)
        assert tnp.device == device.cpu
        assert tnp.numpy().all() == npdata.all()

    def test_to_inplace(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, device=device.cpu)

        t.to_(device.cuda)
        assert t.device == device.cuda
        assert t.numpy().all() == npdata.all()

        t.to_(device.cpu)
        assert t.device == device.cpu
        assert t.numpy().all() == npdata.all()
