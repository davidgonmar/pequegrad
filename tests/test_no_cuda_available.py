import numpy as np
import pytest
from pequegrad.cuda import CUDA_AVAILABLE
from pequegrad.tensor import Tensor


@pytest.mark.skipif(CUDA_AVAILABLE, reason="CUDA is available")
class TestNoCudaAvailable:
    # use fixture force_no_cuda
    def test_throws_trying_to_create_cuda_tensor(self):
        with pytest.raises(NotImplementedError):
            t = Tensor(np.array([1, 2, 3]), storage="cuda")
            t = t.to("cuda")

    def test_throws_trying_to_move_to_cuda(self):
        t = Tensor(np.array([1, 2, 3]), storage="np")
        with pytest.raises(NotImplementedError):
            t.to("cuda")

    def test_throws_trying_to_move_to_cuda_inplace(self):
        t = Tensor(np.array([1, 2, 3]), storage="np")
        with pytest.raises(NotImplementedError):
            t.to_("cuda")
