import pequegrad.ops as ops
from pequegrad.tensor import Tensor
import pequegrad.complex as cplx

PI = 3.14159265358979323846


def fft(x: Tensor) -> Tensor:
    # https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
    assert x.ndim == 1
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = cplx.exp(
            cplx.ComplexTensor.from_imag(
                -2 * PI * ops.arange(0, N, 1, x.dtype, x.device) / N
            )
        )

        X = cplx.cat(
            [
                factor[: int(N / 2)] * X_odd + X_even,
                factor[int(N / 2) :] * X_odd + X_even,
            ]
        )
        return X
