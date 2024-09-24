from pequegrad import Tensor, device
import numpy as np
import pytest


cuda = device.cuda(0)
cpu = device.cpu(0)


@pytest.mark.skipif(True, reason="needs fix")
class TestTo:
    @pytest.mark.parametrize("device", [cpu, cuda])
    def test_device(self, device):
        t = Tensor(np.array([1, 2, 3]), device=device)
        assert t.device == device

    def test_to(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, device=cuda)

        tcu = t.to(cuda)
        assert tcu.device == cuda
        assert tcu.numpy().all() == npdata.all()

        # test that to("cuda") does not modify the original tensor
        assert t.device == cpu
        assert t.numpy().all() == npdata.all()

        tnp = tcu.to(cpu)
        assert tnp.device == cpu
        assert tnp.numpy().all() == npdata.all()

    def test_to_inplace(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, device=cpu)

        t.to_(cuda)
        assert t.device == cuda
        assert t.numpy().all() == npdata.all()

        t.to_(cpu)
        assert t.device == cpu
        assert t.numpy().all() == npdata.all()
