import pequegrad as pg
import numpy as np
import pytest


def _fn1(x):
    return x.log().exp().log().exp()


def _fn2(x):
    return x.log()


def _fn3(x):
    return x.exp()


class TestCompile:
    @pytest.mark.parametrize("fn", [_fn1, _fn2, _fn3])
    def test_compile(self, fn):
        arr1 = np.random.randn(20000).astype(np.float32)
        x = pg.Tensor(arr1).to(pg.device.cuda)
        x_ = pg.Tensor(arr1).to(pg.device.cuda)
        x2 = fn(x)
        x3 = fn(x_)
        pg.compile(x2)
        np.testing.assert_allclose(x2.eval(), x3.eval(), atol=1e-5)
