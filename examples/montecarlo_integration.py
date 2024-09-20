from pequegrad.transforms import montecarlo_integrate, Tensor
import time


def f(x):
    return x**2 + x**3


# integral is 1/3 * x^3 + 1/4 * x^4
# so the integral from 0 to 1 is 1/3 + 1/4 = 7/12 = 0.5833333333333334

integral_fn = montecarlo_integrate(f, n_samples=10000000)


a, b = Tensor(0.0), Tensor(1.0)  # interval [0, 1]

n_samples = 1000

t0 = time.time()

integral = integral_fn(a, b)

t1 = time.time()

print(f"Integral: {integral.numpy()} in {t1 - t0:.4f} seconds")
