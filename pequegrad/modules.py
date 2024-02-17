from pequegrad.tensor import Tensor
from pequegrad.storage import NumpyStorage
import numpy as np
from typing import List
import pickle


class ModuleParam(Tensor):
    pass


def kaiming_init(shape):
    fan_in = shape[0]
    bound = 1 / np.sqrt(fan_in)
    return ModuleParam.uniform(shape, -bound, bound, requires_grad=True)


class Module:
    _parameters: List[Tensor] = None

    @property
    def storage_type(self):
        return self.parameters()[0].storage_type

    def save(self, path):
        params = [p.numpy() for p in self.parameters()]
        d = {"params": params}
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
            loaded_params = []
            orig_storage_type = self.parameters()[0].storage_type
            for p, p_loaded in zip(self.parameters(), d["params"]):
                p.data = NumpyStorage(p_loaded)  # always load to numpy (cpu) by default

            self._parameters = loaded_params  # force re-creation of the parameters list
            self.to("cuda" if orig_storage_type == "cuda" else "np")

    def to(self, storage_type):
        for p in self.parameters():
            p.to_(storage_type)
        return self

    def _search_parameters(self):
        params = []
        for p in self.__dict__.values():
            if isinstance(p, ModuleParam):
                params.append(p)
            elif isinstance(p, Module):
                params.extend(p._search_parameters())
        return params

    def parameters(self):
        # first call to parameters, we need to retrieve them, then they are cached
        if not self._parameters:
            self._parameters = self._search_parameters()
        return self._parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = kaiming_init((in_features, out_features))
        self.bias = ModuleParam.zeros(out_features, requires_grad=True)

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
        self.bias = ModuleParam.zeros(out_channels, requires_grad=True)
        assert stride == 1, "only stride=1 is supported"
        assert padding == 0, "only padding=0 is supported"

    def forward(self, input):
        return input.conv2d(self.kernel, self.bias)
