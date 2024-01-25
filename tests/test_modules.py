from pequegrad.tensor import Tensor
from pequegrad.modules import Linear, Conv2d


class TestModules:
    def test_linear(self):
        l = Linear(2, 1)
        x = Tensor([1.0, 2.0])
        y = l.forward(x)
        assert y.shape == (1,)

    def test_conv2d(self):
        c = Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        x = Tensor.ones((1, 1, 3, 3))
        y = c.forward(x)
        assert y.shape == (1, 1, 2, 2)
