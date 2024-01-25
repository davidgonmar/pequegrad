from pequegrad.tensor import Tensor
import numpy as np
from typing import List


def kaiming_init(n, shape):
    return Tensor(np.random.normal(0, np.sqrt(1 / n), shape), requires_grad=True)


class Module:
    _parameters: List[Tensor] = []

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weights = kaiming_init(in_features, (in_features, out_features))
        self.bias = Tensor.zeros(out_features, requires_grad=True)
        self._parameters = [self.weights, self.bias]

    def forward(self, input):
        return (input @ self.weights) + self.bias

    def backward(self, output_grad: Tensor):
        self.weights.backward(output_grad)
        self.bias.backward(output_grad)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.kernel = kaiming_init(
            in_channels, (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = Tensor.zeros((out_channels,), requires_grad=True)
        self._parameters = [self.kernel, self.bias]
        assert stride == 1, "only stride=1 is supported"
        assert padding == 0, "only padding=0 is supported"

    def forward(self, input):
        return input.conv2d(self.kernel) + self.bias

    def backward(self, output_grad: Tensor):
        self.kernel.backward(output_grad)
        self.bias.backward(output_grad)
