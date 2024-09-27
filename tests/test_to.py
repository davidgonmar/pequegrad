from pequegrad import Tensor, device, fngrad
import numpy as np
import pytest


cuda = device.cuda(0)
cpu = device.cpu(0)


class TestTo:
    @pytest.mark.parametrize("device", [cpu, cuda])
    def test_device(self, device):
        t = Tensor(np.array([1, 2, 3]), device=device)
        assert t.device == device

    def test_to(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, device=cpu)

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

    def test_to_differentiable(self):
        tcpu = Tensor(np.array([1, 2, 3]), device=cpu)

        def f(t):
            print(t.to(cuda))
            print(t.to(cuda).sum())
            return t.to(cuda).sum()

        tcuda, (grad,) = fngrad(f, wrt=[0], return_outs=True)(tcpu)

        assert tcuda.device == cuda
        assert tcuda.numpy().all() == np.array([1, 2, 3]).all()
        assert grad.device == cpu
        assert grad.numpy().all() == np.array([1, 1, 1]).all()
