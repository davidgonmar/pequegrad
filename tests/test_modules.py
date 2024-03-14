from pequegrad.tensor import Tensor, CUDA_AVAILABLE
from pequegrad.modules import Linear, Conv2d, StatefulModule
import os
import tempfile
import pytest
import numpy as np


class CustomModule(StatefulModule):
    def __init__(self):
        self.a = Linear(2, 1)
        self.b = Linear(2, 1)


class CustomConvModule(StatefulModule):
    def __init__(self):
        self.a = Conv2d(1, 3, 2)
        self.b = Conv2d(1, 3, 2)


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
        m2 = Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            m.save(path)
            m2.load(path)
            assert m.parameters()[0].backend == "np"
            assert m2.parameters()[0].backend == "np"
            for p1, p2 in zip(m.parameters(), m2.parameters()):
                np.testing.assert_allclose(p1.numpy(), p2.numpy())

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
    def test_save_load_cuda(self):
        m = Linear(2, 1).to("cuda")
        m2 = Linear(2, 1).to("cuda")
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            m.save(path)
            m2.load(path)
            assert m.parameters()[0].backend == "cuda"
            assert m2.parameters()[0].backend == "cuda"
            for p1, p2 in zip(m.parameters(), m2.parameters()):
                np.testing.assert_allclose(p1.numpy(), p2.numpy())

    @pytest.mark.parametrize("module", [CustomModule, CustomConvModule])
    def test_find_params_on_custom_module(self, module):
        m = module()
        assert len(m.parameters())  # 2 weights and 2 biases

    @pytest.mark.parametrize("module", [CustomModule, CustomConvModule])
    def test_save_load_custom_module(self, module):
        m = module()
        m2 = module()
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            m.save(path)
            m2.load(path)

            for p1, p2 in zip(m.parameters(), m2.parameters()):
                np.testing.assert_allclose(p1.numpy(), p2.numpy())

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
    def test_to(self):
        m = Linear(2, 1)
        # simulate copy until implemented
        m2 = Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.pkl")
            m.save(path)
            m2.load(path)
            m2.to("cuda")
            for p1, p2 in zip(m.parameters(), m2.parameters()):
                assert p1.backend == "np"
                assert p2.backend == "cuda"
                np.testing.assert_allclose(p1.numpy(), p2.numpy())
        m2.to("cuda")
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert p1.backend == "np"
            assert p2.backend == "cuda"
            np.testing.assert_allclose(p1.numpy(), p2.numpy())

        m2.to("np")
        for p1, p3 in zip(m.parameters(), m2.parameters()):
            assert p1.backend == "np"
            assert p3.backend == "np"
            np.testing.assert_allclose(p1.numpy(), p3.numpy())

        m2.to("cuda")
        for p1, p4 in zip(m.parameters(), m2.parameters()):
            assert p1.backend == "np"
            assert p4.backend == "cuda"
            np.testing.assert_allclose(p1.numpy(), p4.numpy())
