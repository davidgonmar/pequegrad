from pequegrad.tensor import Tensor, CUDA_AVAILABLE
import numpy as np
import pytest


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
class TestTo:
    @pytest.mark.parametrize("backend", ["np", "cuda"])
    def test_backend(self, backend):
        t = Tensor(np.array([1, 2, 3]), backend=backend)
        assert t.backend == backend

    @pytest.mark.parametrize("backend", ["np", "cuda"])
    def test_backend_requires_grad(self, backend):
        t = Tensor(np.array([1, 2, 3]), backend=backend, requires_grad=True)
        assert t.backend == backend
        assert t.requires_grad

    def test_to(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, backend="np")

        tcu = t.to("cuda")
        assert tcu.backend == "cuda"
        assert tcu.numpy().all() == npdata.all()

        # test that to("cuda") does not modify the original tensor
        assert t.backend == "np"
        assert t.numpy().all() == npdata.all()

        tnp = tcu.to("np")
        assert tnp.backend == "np"
        assert tnp.numpy().all() == npdata.all()

    def test_to_requires_grad(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, backend="np", requires_grad=True)

        tcu = t.to("cuda")
        assert tcu.backend == "cuda"
        assert tcu.requires_grad

        # test that to("cuda") does not modify the original tensor
        assert t.backend == "np"
        assert t.requires_grad

        tnp = tcu.to("np")
        assert tnp.backend == "np"
        assert tnp.requires_grad

    def test_to_inplace(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, backend="np")

        t.to_("cuda")
        assert t.backend == "cuda"
        assert t.numpy().all() == npdata.all()

        t.to_("np")
        assert t.backend == "np"
        assert t.numpy().all() == npdata.all()

    def test_to_inplace_requires_grad(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, backend="np", requires_grad=True)

        t.to_("cuda")
        assert t.backend == "cuda"
        assert t.requires_grad
        t.to_("np")
        assert t.requires_grad
