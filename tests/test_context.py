from pequegrad.tensor import Tensor, Add
from pequegrad.context import no_grad


class TestContext:
    def test_graph_build_grad(self):
        x = Tensor(1.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x + y

        assert isinstance(z._ctx, Add)
        assert z.grad.data.numpy() == 0.0

    def test_graph_build_no_grad(self):
        with no_grad():
            x = Tensor(1.0, requires_grad=True)
            y = Tensor(2.0, requires_grad=True)
            z = x + y

        assert z.grad is None
