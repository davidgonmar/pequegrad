class ComplexTensor:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexTensor(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexTensor(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
