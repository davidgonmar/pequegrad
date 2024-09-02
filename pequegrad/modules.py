from pequegrad.backend.c import Tensor, device
import numpy as np
from typing import List, Union
import pickle
from pequegrad.context import pequegrad_context


class ModuleParam(Tensor):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], Tensor):
            super().__init__(args[0].numpy(), *args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)


def kaiming_init(shape):
    fan_in = shape[0]
    bound = 1 / np.sqrt(fan_in)
    uniform = np.random.uniform(low=-bound, high=bound, size=shape).astype(np.float32)
    return ModuleParam(uniform)


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
            device = self.parameters()[0].device
            for p, p_loaded in zip(self.parameters(), d["params"]):
                p.assign(Tensor(p_loaded, device=device))

    def to(self, backend):
        d = {"cpu": device.cpu, "cuda": device.cuda}
        backend = d[backend] if backend in d else backend
        for p in self.__dict__.values():
            if isinstance(p, StatefulModule) or isinstance(p, NonStatefulModule):
                p.to(backend)
            elif isinstance(p, ModuleParam):
                p.to_(backend)
            elif isinstance(p, (list, tuple)):
                for pp in p:
                    if isinstance(pp, StatefulModule) or isinstance(
                        pp, NonStatefulModule
                    ):
                        pp.to(backend)
                    elif isinstance(pp, ModuleParam):
                        pp.to_(backend)
        return self

    def _search_parameters_and_submodules(self, root=False):
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
        if root:
            self._parameters = params
            self._submodules = submodules

        return params, submodules

    def parameters(self):
        # first call to parameters, we need to retrieve them, then they are cached
        if not self._parameters:
            self._search_parameters_and_submodules(root=True)
        return self._parameters

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(StatefulModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = kaiming_init((in_features, out_features))
        self.bias = ModuleParam.zeros((out_features,)) if bias else None

    def forward(self, input):
        a = input @ self.weight
        if self.bias is not None:
            a += self.bias
        return a


class Conv2d(StatefulModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel = kaiming_init(
            (out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = ModuleParam.zeros((out_channels,)) if bias else None
        assert isinstance(self.kernel, ModuleParam)
        assert isinstance(self.bias, ModuleParam) or self.bias is None
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

    def to(self, backend):
        for p in self.__dict__.values():
            if isinstance(p, StatefulModule) or isinstance(p, NonStatefulModule):
                p.to(backend)
            elif isinstance(p, ModuleParam):
                p.to_(backend)

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


class Embedding(StatefulModule):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = kaiming_init((num_embeddings, embedding_dim))

    def forward(self, input: Tensor) -> Tensor:
        return self.weight[input]

    def with_slices(self, start: int, end: int, stride: int = 1):
        return self.weight[start:end:stride]


class LayerNorm(StatefulModule):
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = kaiming_init(
            normalized_shape
            if isinstance(normalized_shape, (tuple, list))
            else (normalized_shape,)
        )
        self.bias = ModuleParam.zeros((normalized_shape,))

    def forward(self, input: Tensor) -> Tensor:
        return input.layer_norm(self.normalized_shape, self.eps, self.weight, self.bias)


class GELU(StatefulModule):
    def forward(self, input: Tensor) -> Tensor:
        return input.gelu()


class ModuleList(StatefulModule):
    def __init__(self, modules):
        for i, module in enumerate(modules):
            setattr(self, str(i), module)
        self.keys = [str(i) for i in range(len(modules))]

    # iter and getitem
    def __iter__(self):
        for key in self.keys:
            yield getattr(self, key)

    def __getitem__(self, idx):
        return getattr(self, self.keys[idx])

    def __len__(self):
        return len(self.keys)

    # forward
    def forward(self, input: Tensor) -> Tensor:
        for key in self.keys:
            input = getattr(self, key)(input)
        return input


# like pytorch
class ModuleDict(StatefulModule):
    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError


Module = StatefulModule
