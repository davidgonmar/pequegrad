from pequegrad.tensor import Tensor
from typing import Tuple
from .function import Function
import numpy as np


class Unsqueeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            self.a.data.expand_dims(axis=self.dim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.squeeze(axis=self.dim), storage=self.storage
            )


class Squeeze(Function):
    def __init__(self, a: Tensor, dim: int):
        super().__init__(a)
        self.a = a
        self.dim = dim

    def forward(self):
        self.ret = Tensor(
            self.a.data.squeeze(axis=self.dim),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.expand_dims(axis=self.dim), storage=self.storage
            )


class Permute(Function):
    def __init__(self, a: Tensor, dims: Tuple[int, ...]):
        super().__init__(a)
        self.a = a
        self.dims = dims

    def forward(self):
        self.ret = Tensor(
            self.a.data.permute(*self.dims),
            requires_grad=self.requires_grad,
            storage=self.a.device,
        )

        return self.ret

    def backward(self):
        bw_dims = np.argsort(
            self.dims
        )  # computes the indices that would sort the dims back
        if self.a.requires_grad:
            self.a._grad += Tensor(
                self.ret.grad.data.permute(*bw_dims), storage=self.storage
            )


class Reshape(Function):
    def __init__(self, input: Tensor, shape: Tuple[int, ...]):
        super().__init__(input)
        self.input = input
        self.output_shape = shape
        self.input_shape = input.shape

    def forward(self) -> Tensor:
        self.ret = Tensor(
            self.input.data.reshape(self.output_shape),
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            self.input._grad += Tensor(
                self.ret.grad.data, storage=self.storage
            ).reshape(self.input_shape)
