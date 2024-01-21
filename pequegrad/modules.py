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
        self.weights = kaiming_init(in_features, (out_features, in_features))
        self.bias = Tensor.zeros(out_features, requires_grad=True)

        self._parameters = [self.weights, self.bias]

    def forward(self, input):
        return (input @ self.weights.transpose()) + self.bias

    def backward(self, output_grad: Tensor):
        self.weights.backward(output_grad)
        self.bias.backward(output_grad)
