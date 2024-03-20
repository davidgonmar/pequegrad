from pequegrad.tensor import Tensor
from typing import Tuple, Union
from .function import Function, BackendTensor


class Unfold(Function):
    def forward(
        self,
        input: BackendTensor,
        kernel_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
    ) -> Tensor:
        unfolded = input.im2col(kernel_shape, stride=stride, dilation=dilation)
        if self.requires_grad:
            self.input = input
            self.kernel_shape = kernel_shape
            self.stride = stride
            self.dilation = dilation
        return unfolded

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.requires_grad:
            folded_grad = grad_output.col2im(
                self.kernel_shape,
                self.input.shape[-2:],
                stride=self.stride,
                dilation=self.dilation,
            )
            return folded_grad


class Fold(Function):
    def forward(
        self,
        input: BackendTensor,
        kernel_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
    ) -> Tensor:
        if self.requires_grad:
            self.input = input
            self.kernel_shape = kernel_shape
            self.stride = stride
            self.dilation = dilation
        return input.col2im(
            kernel_shape,
            output_shape,
            stride=stride,
            dilation=dilation,
        )

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.needs_input_grad[0]:
            unfolded = grad_output.im2col(
                self.kernel_shape, stride=self.stride, dilation=self.dilation
            )
            return unfolded
        return None
