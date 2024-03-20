from typing import Tuple
from .function import Function, BackendTensor
import numpy as np


class Unsqueeze(Function):
    def forward(self, a: BackendTensor, dim: int) -> BackendTensor:
        self.dim = dim
        return a.expand_dims(axis=dim)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            return grad_output.squeeze(axis=self.dim)


class Squeeze(Function):
    def forward(self, a: BackendTensor, dim: int) -> BackendTensor:
        self.dim = dim
        return a.squeeze(axis=dim)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            return grad_output.expand_dims(axis=self.dim)


class Permute(Function):
    def forward(self, a: BackendTensor, dims: Tuple[int, ...]) -> BackendTensor:
        self.dims = dims
        return a.permute(*dims)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            bw_dims = np.argsort(
                self.dims
            )  # computes the indices that would sort the dims back
            return grad_output.permute(*bw_dims)


class Reshape(Function):
    def forward(self, input: BackendTensor, shape: Tuple[int, ...]) -> BackendTensor:
        self.input_shape = input.shape
        return input.reshape(*shape)

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            return grad_output.reshape(*self.input_shape)
