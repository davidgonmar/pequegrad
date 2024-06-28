from pequegrad import Tensor, device, dt
from pequegrad.compile import jit
import numpy as np
import time

dev = None


def test_some_fn():
    @jit
    def some_function(x, y, z):
        return x.log() + y + z.exp().log().exp()

    def non_jitted(x, y, z):
        return x.log() + y + z.exp().log().exp()

    for i in range(10):
        x = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()
        y = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()
        z = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32).eval()

        start = time.time()
        j = some_function(x, y, z).eval()
        jittedtime = time.time() - start

        start = time.time()
        nj = non_jitted(x, y, z).eval()
        nonjittedtime = time.time() - start

        print(f"Jitted time: {jittedtime}, Non-jitted time: {nonjittedtime}")
        np.testing.assert_allclose(j.numpy(), nj.numpy(), atol=1e-3)


def test_relu():
    @jit
    def some_function(x):
        return x.relu()

    def non_jitted(x):
        return x.relu()

    for i in range(10):
        x = Tensor(np.random.randn(10000, 1000), device=dev).astype(dt.float32)
        start = time.time()
        nj = non_jitted(x).eval()
        nonjittedtime = time.time() - start

        start = time.time()
        j = some_function(x).eval()
        jittedtime = time.time() - start

        print(f"Jitted time: {jittedtime}, Non-jitted time: {nonjittedtime}")
        np.testing.assert_allclose(j.numpy(), nj.numpy(), atol=1e-3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", type=str, default="some_fn", help="Test to run")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computations")

    args = parser.parse_args()

    if args.cuda:
        dev = device.cuda
    else:
        dev = device.cpu

    if args.test == "some_fn":
        test_some_fn()
    elif args.test == "relu":
        test_relu()
    else:
        raise ValueError(f"Test {args.test} not found")
