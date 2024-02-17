from pequegrad.tensor import Tensor
import numpy as np
from typing import List
import pickle


def kaiming_init(shape):
    fan_in = shape[0]
    bound = 1 / np.sqrt(fan_in)
    return Tensor.uniform(shape, -bound, bound, requires_grad=True)


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model._topickle(), f)


def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Module._frompickle(data)


class Module:
    _parameters: List[Tensor] = []

    def _topickle(self):
        st = self.parameters()[0].storage_type
        mod = self.to("np")
        return {"module": mod, "actual_storage": st}

    @staticmethod
    def _frompickle(data):
        cl = data["module"]
        return cl.to(data["actual_storage"])

    def to(self, storage_type):
        for p in self._parameters:
            p.to_(storage_type)
        return self

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = kaiming_init((in_features, out_features))
        self.bias = Tensor.zeros(out_features, requires_grad=True)
        self._parameters = [self.weights, self.bias]

    def forward(self, input):
        return (input @ self.weights) + self.bias

    def backward(self, output_grad: Tensor):
        self.weights.backward(output_grad)
        self.bias.backward(output_grad)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel = kaiming_init(
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = Tensor.zeros(out_channels, requires_grad=True)
        self._parameters = [self.kernel, self.bias]
        assert stride == 1, "only stride=1 is supported"
        assert padding == 0, "only padding=0 is supported"

    def forward(self, input):
        return input.conv2d(self.kernel, self.bias)
