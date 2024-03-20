from .function import Function, BackendTensor


class Pow(Function):
    def forward(self, base: BackendTensor, exponent: BackendTensor):
        self.base, self.exponent = (
            (base, exponent) if self.requires_grad else (None, None)
        )
        ret = base.power(exponent)
        if self.requires_grad:
            self.ret = ret
        return ret

    def backward(self, grad_output: BackendTensor):
        base_grad, exponent_grad = None, None
        if self.needs_input_grad[0]:
            base_grad = grad_output * self.exponent * self.base.power(self.exponent - 1)
        if self.needs_input_grad[1]:
            exponent_grad = grad_output * self.ret * self.base.log()

        return base_grad, exponent_grad


class Log(Function):
    def forward(self, a: BackendTensor):
        self.a = a if self.requires_grad else None
        return a.log()

    def backward(self, grad_output: BackendTensor):
        if self.requires_grad:
            return grad_output / self.a
        return None


class Exp(Function):
    def forward(self, a: BackendTensor):
        ret = a.exp()
        if self.requires_grad:
            self.a = a
            self.ret = ret
        return ret

    def backward(self, grad_output: BackendTensor):
        if self.requires_grad:
            return grad_output * self.ret
        return None


class ReLU(Function):
    """Implements the ReLU activation function: ReLU(x) = max(0, x)"""

    def forward(self, a: BackendTensor):
        if self.requires_grad:
            self.a = a

        return a.el_wise_max(0)

    def backward(self, grad_output: BackendTensor):
        # grad = 1 if a > 0 else 0
        if self.requires_grad:
            concrete_class = self.a.__class__
            return concrete_class.where_static(self.a > 0, 1, 0) * grad_output
