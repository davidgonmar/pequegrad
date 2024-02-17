from pequegrad.tensor import Tensor, CUDA_AVAILABLE
from pequegrad.modules import Linear, Conv2d, save_model, load_model
import os
import tempfile
import pytest
import numpy as np


class TestModules:
    def test_linear(self):
        li = Linear(2, 1)
        x = Tensor([1.0, 2.0])
        y = li.forward(x)
        assert y.shape == (1,)

    def test_conv2d(self):
        c = Conv2d(in_channels=1, out_channels=3, kernel_size=2)
        x = Tensor.ones((1, 1, 3, 3))
        y = c.forward(x)
        assert y.shape == (1, 3, 2, 2)

    def test_save_load_np(self):
        m = Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            save_model(m, path)
            m2 = load_model(path)

            for p1, p2 in zip(m.parameters(), m2.parameters()):
                np.testing.assert_allclose(p1.numpy(), p2.numpy())

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
    def test_save_load_cuda(self):
        m = Linear(2, 1).to("cuda")
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            save_model(m, path)
            m2 = load_model(path)

            for p1, p2 in zip(m.parameters(), m2.parameters()):
                np.testing.assert_allclose(p1.numpy(), p2.numpy())
