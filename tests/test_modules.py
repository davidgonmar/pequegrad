from pequegrad.tensor import Tensor
from pequegrad.modules import Linear


class TestModules:
    def test_linear(self):
        l = Linear(2, 1)
        x = Tensor([1.0, 2.0])
        y = l.forward(x)
        assert y.shape == (1,)
