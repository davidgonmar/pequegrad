from pequegrad.tensor import Tensor
from pequegrad.backend import NumpyTensor, CudaTensor
import numpy as np
from typing import List, Union
import pickle
from pequegrad.context import pequegrad_context


class ModuleParam(Tensor):
    pass


def kaiming_init(shape):
    fan_in = shape[0]
    bound = 1 / np.sqrt(fan_in)
    return ModuleParam.uniform(shape, -bound, bound, requires_grad=True)


class StatefulModule:
    _parameters: List[ModuleParam] = None
    _training: bool = True
    _submodules: List["StatefulModule"] = None

    @property
    def backend(self):
        return self.parameters()[0].backend if len(self.parameters()) > 0 else None

    def _propagate_training(self):
        if self._submodules is None:
            self._search_parameters_and_submodules()
        for m in self._submodules:
            m._training = self._training
            m._propagate_training()

    def train(self):
        self._training = True
        self._propagate_training()

    def eval(self):
        self._training = False
        self._propagate_training()

    @property
    def training(self):
        if pequegrad_context.force_training is not None:
            return pequegrad_context.force_training
        return self._training

    def save(self, path):
        params = [p.numpy() for p in self.parameters()]
        d = {"params": params}
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
            orig_backend = self.parameters()[0].backend
            for p, p_loaded in zip(self.parameters(), d["params"]):
                p.assign(
                    NumpyTensor(p_loaded)
                    if orig_backend == "np"
                    else CudaTensor(p_loaded)
                )

    def to(self, backend):
        for p in self.parameters():
            p.to_(backend)
        return self

    def _search_parameters_and_submodules(self):
        params = []
        submodules = []
        for p in self.__dict__.values():
            if isinstance(p, ModuleParam):
                params.append(p)
            elif isinstance(p, StatefulModule):
                params.extend(p._search_parameters_and_submodules()[0])
                submodules.append(p)
            elif isinstance(p, (list, tuple)):  # search in lists and tuples
                for pp in p:
                    if isinstance(pp, StatefulModule):
                        params.extend(pp._search_parameters_and_submodules()[0])
                        submodules.append(pp)
                    elif isinstance(pp, ModuleParam):
                        params.append(pp)

        self._parameters = params
        self._submodules = submodules

        return params, submodules

    def parameters(self):
        # first call to parameters, we need to retrieve them, then they are cached
        if not self._parameters:
            self._search_parameters_and_submodules()
        return self._parameters

    def reset_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(StatefulModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = kaiming_init((in_features, out_features))
        self.bias = ModuleParam.zeros(out_features, requires_grad=True)

    def forward(self, input):
        a = (input @ self.weights) + self.bias
        return a

    def backward(self, output_grad: Tensor):
        self.weights.backward(output_grad)
        self.bias.backward(output_grad)


class Conv2d(StatefulModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super().__init__()
        self.kernel = kaiming_init(
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = ModuleParam.zeros(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input: Tensor) -> Tensor:
        return input.conv2d(
            self.kernel,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class NonStatefulModule:
    def forward(self, input):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MaxPool2d(NonStatefulModule):
    def __init__(
        self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None
    ):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return input.max_pool2d(self.kernel_size, self.stride)


class ReLU(NonStatefulModule):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class Reshape(NonStatefulModule):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        return input.reshape(self.shape)


class LocalResponseNorm(NonStatefulModule):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=2):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return input.local_response_norm(self.size, self.alpha, self.beta, self.k)


class Sequential(StatefulModule):
    def __init__(self, *args: Union[StatefulModule, NonStatefulModule]):
        super().__init__()
        self.modules = args

    def forward(self, input: Tensor) -> Tensor:
        for module in self.modules:
            input = module.forward(input)
        return input


class Dropout(StatefulModule):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return input.dropout(self.p, self.training)
