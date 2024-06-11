import pequegrad as pg
import numpy as np
import pytest


def _fn1(x):
    return x.log().exp().log().exp()


def _fn2(x):
    return x.log()


def _fn3(x):
    return x.exp()


def _fn4(x, y):
    return x + y.exp()


class TestCompile:
    @pytest.mark.parametrize("fn", [_fn4])
    def test_compile2(self, fn):
        arr1 = np.random.randn(200).astype(np.float32)
        arr2 = np.random.randn(200).astype(np.float32)
        x = pg.Tensor(arr1).to(pg.device.cuda)
        y = pg.Tensor(arr2).to(pg.device.cuda)
        x_ = pg.Tensor(arr1).to(pg.device.cuda)
        y_ = pg.Tensor(arr2).to(pg.device.cuda)
        x2 = fn(x, y)
        x3 = fn(x_, y_)
        pg.compile(x2)
        np.testing.assert_allclose(x2.numpy(), x3.numpy(), atol=1e-5)

    @pytest.mark.parametrize("fn", [_fn1, _fn2, _fn3])
    def test_compile(self, fn):
        arr1 = np.random.randn(200).astype(np.float32)
        x = pg.Tensor(arr1).to(pg.device.cuda)
        x_ = pg.Tensor(arr1).to(pg.device.cuda)
        x2 = fn(x)
        x3 = fn(x_)
        pg.compile(x2)
        np.testing.assert_allclose(x2.numpy(), x3.numpy(), atol=1e-5)
