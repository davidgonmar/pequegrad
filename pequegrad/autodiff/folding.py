from pequegrad.tensor import Tensor
from typing import Tuple, Union
from .function import Function


class Unfold(Function):
    def __init__(
        self,
        input: Tensor,
        kernel_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, ...]],
        dilation: Union[int, Tuple[int, ...]],
    ):
        super().__init__(input)
        self.input = input
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.dilation = dilation

    def forward(self) -> Tensor:
        unfolded = self.input.data.im2col(
            self.kernel_shape, stride=self.stride, dilation=self.dilation
        )
        self.ret = Tensor(
            unfolded,
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            folded_grad = self.ret.grad.data.col2im(
                self.kernel_shape,
                self.input.shape[-2:],
                stride=self.stride,
                dilation=self.dilation,
            )
            self.input._grad += Tensor(folded_grad, storage=self.storage)


class Fold(Function):
    def __init__(
        self,
        input: Tensor,
        kernel_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
    ):
        super().__init__(input)
        self.input = input
        self.kernel_shape = kernel_shape
        self.output_shape = output_shape
        self.stride = stride
        self.dilation = dilation

    def forward(self) -> Tensor:
        folded = self.input.data.col2im(
            self.kernel_shape,
            self.output_shape,
            stride=self.stride,
            dilation=self.dilation,
        )
        self.ret = Tensor(
            folded,
            requires_grad=self.requires_grad,
            storage=self.input.device,
        )
        return self.ret

    def backward(self) -> Tensor:
        if self.input.requires_grad:
            unfolded = self.ret.grad.data.im2col(
                self.kernel_shape, stride=self.stride, dilation=self.dilation
            )
            self.input._grad += Tensor(unfolded, storage=self.storage)
