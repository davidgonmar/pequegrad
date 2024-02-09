from pequegrad.tensor import Tensor, CUDA_AVAILABLE
import numpy as np
import pytest


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
class TestTo:
    @pytest.mark.parametrize("storage", ["np", "cuda"])
    def test_storage_type(self, storage):
        t = Tensor(np.array([1, 2, 3]), storage=storage)
        assert t.storage_type == storage

    @pytest.mark.parametrize("storage", ["np", "cuda"])
    def test_storage_type_requires_grad(self, storage):
        t = Tensor(np.array([1, 2, 3]), storage=storage, requires_grad=True)
        assert t.storage_type == storage
        assert t.requires_grad
        assert t.grad.storage_type == storage

    def test_to(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, storage="np")

        tcu = t.to("cuda")
        assert tcu.storage_type == "cuda"
        assert tcu.numpy().all() == npdata.all()

        # test that to("cuda") does not modify the original tensor
        assert t.storage_type == "np"
        assert t.numpy().all() == npdata.all()

        tnp = tcu.to("np")
        assert tnp.storage_type == "np"
        assert tnp.numpy().all() == npdata.all()

    def test_to_requires_grad(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, storage="np", requires_grad=True)

        tcu = t.to("cuda")
        assert tcu.storage_type == "cuda"
        assert tcu.requires_grad
        assert tcu.grad.storage_type == "cuda"

        # test that to("cuda") does not modify the original tensor
        assert t.storage_type == "np"
        assert t.requires_grad
        assert t.grad.storage_type == "np"

        tnp = tcu.to("np")
        assert tnp.storage_type == "np"
        assert tnp.requires_grad
        assert tnp.grad.storage_type == "np"

    def test_to_inplace(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, storage="np")

        t.to_("cuda")
        assert t.storage_type == "cuda"
        assert t.numpy().all() == npdata.all()

        t.to_("np")
        assert t.storage_type == "np"
        assert t.numpy().all() == npdata.all()

    def test_to_inplace_requires_grad(self):
        npdata = np.array([1, 2, 3])
        t = Tensor(npdata, storage="np", requires_grad=True)

        t.to_("cuda")
        assert t.storage_type == "cuda"
        assert t.requires_grad
        assert t.grad.storage_type == "cuda"

        t.to_("np")
        assert t.storage_type == "np"
        assert t.requires_grad
        assert t.grad.storage_type == "np"
