import pytest
from pequegrad import custom_prim, Tensor, grads
import numpy as np


@custom_prim
def myfunction(x, y):
    return x + y


@myfunction.vjp
def myfunction_vjp(primals, tangents, outputs):
    x, y = primals
    return x, y


@pytest.mark.parametrize(
    "a_values, b_values, expected_c, expected_g0, expected_g1",
    [
        (
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 5.0, 7.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ),
        (
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        (
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
        ),
    ],
)
def test_myfunction(a_values, b_values, expected_c, expected_g0, expected_g1):
    a = Tensor(np.array(a_values))
    b = Tensor(np.array(b_values))

    c = myfunction(a, b)
    g = grads([a, b], c)

    assert np.allclose(c.numpy(), expected_c)
    assert np.allclose(g[0].numpy(), expected_g0)
    assert np.allclose(g[1].numpy(), expected_g1)


@custom_prim
def myfunction2(x, y, z, w):
    return (x.log() + y.exp() / z).relu() * w


@myfunction2.vjp
def myfunction2_vjp(primals, tangents, outputs):
    out = outputs[0]
    x, y, z, w = primals
    outg = tangents[0]
    return x * 2 * out + outg, y * x, y.log() * outg, z * 2 + out * outg.log()


@pytest.mark.parametrize("shape", [(3, 4), (5, 6)])
def test_myfunction2(shape):
    x = Tensor(np.random.rand(*shape))
    y = Tensor(np.random.rand(*shape))
    z = Tensor(np.random.rand(*shape))
    w = Tensor(np.random.rand(*shape))

    tan = Tensor(np.random.rand(*shape))

    c = myfunction2(x, y, z, w)
    print("c", c)

    g = grads([x, y, z, w], c, tan)

    cnumpy = (
        np.maximum(0, np.log(x.numpy()) + np.exp(y.numpy()) / z.numpy()) * w.numpy()
    )
    assert np.allclose(c.numpy(), cnumpy)
    np.testing.assert_allclose(g[0].numpy(), 2 * x.numpy() * c.numpy() + tan.numpy())
    np.testing.assert_allclose(g[1].numpy(), x.numpy() * y.numpy())
    np.testing.assert_allclose(g[2].numpy(), np.log(y.numpy()) * tan.numpy())
    np.testing.assert_allclose(
        g[3].numpy(), 2 * z.numpy() + c.numpy() * np.log(tan.numpy())
    )
