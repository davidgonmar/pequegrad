import pequegrad.ops as ops


def binop_wrapper(op):
    def wrapped(self, other):
        other = (
            other
            if isinstance(other, ComplexTensor)
            else ComplexTensor.from_real(other)
        )
        return op(self, other)

    return wrapped


class ComplexTensor:
    def __init__(self, real, imag):
        assert isinstance(real, ops.Tensor)
        assert isinstance(imag, ops.Tensor)
        assert real.shape == imag.shape
        self.real = real
        self.imag = imag

    @staticmethod
    def from_real(x):
        return ComplexTensor(x, ops.fill(x.shape, x.dtype, 0, x.device))

    @staticmethod
    def from_imag(x):
        return ComplexTensor(ops.fill(x.shape, x.dtype, 0, x.device), x)

    @binop_wrapper
    def __add__(self, other):
        return ComplexTensor(self.real + other.real, self.imag + other.imag)

    @binop_wrapper
    def __mul__(self, other):
        return ComplexTensor(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __getitem__(self, idx):
        return ComplexTensor(self.real[idx], self.imag[idx])

    def __repr__(self):
        return f"ComplexTensor({self.real}, {self.imag})"

    def numpy(self):
        return self.real.numpy() + 1j * self.imag.numpy()

    @property
    def shape(self):
        return self.real.shape


def exp(x: ComplexTensor) -> ComplexTensor:
    return ComplexTensor(
        ops.exp(x.real) * ops.cos(x.imag),
        ops.exp(x.real) * ops.sin(x.imag),
    )


def cat(x: list) -> ComplexTensor:
    return ComplexTensor(ops.cat([t.real for t in x]), ops.cat([t.imag for t in x]))
