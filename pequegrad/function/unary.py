from typing import Union
from pequegrad.tensor import Tensor
from pequegrad.function import Function

class Pow(Function):
    def __init__(self, base: Tensor, exponent: Union[float, int, Tensor]):
        self.base = base
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent, requires_grad=False, storage=self.base.device)
        self.exponent = exponent

        super().__init__(base, exponent)

    def forward(self):
        self.ret = Tensor(
            self.base.data.power(self.exponent.data),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        if self.base.requires_grad:
            self.base._grad += Tensor(
                self.ret.grad.data
                * self.exponent.data
                * self.base.data.power(self.exponent.data - 1),
                storage=self.storage,
            )
        if self.exponent.requires_grad:
            self.exponent._grad += Tensor(
                self.ret.grad.data * self.ret.data * self.base.data.log(),
                storage=self.storage,
            )


class Log(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a

    def forward(self):
        self.ret = Tensor(
            self.a.data.log(),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad / self.a.data


class Exp(Function):
    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a

    def forward(self):
        self.ret = Tensor(
            self.a.data.exp(),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )

    def backward(self):
        if self.a.requires_grad:
            self.a._grad += self.ret.grad * Tensor(self.ret.data, storage=self.storage)


class ReLU(Function):
    """Implements the ReLU activation function: ReLU(x) = max(0, x)"""

    def __init__(self, a: Tensor):
        super().__init__(a)
        self.a = a
        self.ret: Tensor = None

    def forward(self):
        self.ret = Tensor(
            self.a.data.el_wise_max(0),
            requires_grad=self.requires_grad,
            storage=self.storage,
        )
        return self.ret

    def backward(self):
        # grad = 1 if a > 0 else 0
        if self.a.requires_grad:
            concrete_class = self.a.data.__class__
            self.a._grad += (
                Tensor(
                    # steal the concrete class from tensor storage
                    concrete_class.where_static(self.a.data > 0, 1, 0),
                    requires_grad=False,
                    storage=self.storage,
                )
                * self.ret.grad
            )
