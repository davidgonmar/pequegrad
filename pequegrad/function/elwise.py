from .function import Function
from pequegrad.tensor import Tensor




class ElWiseFunction(Function):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y)
        self.x = x
        self.y = y
    
    def _unbroadcast(self, grad_output, input_shape):
        # If, for example, x was shape (200) and y was shape (32, 200), in the forward pass we "broadcasted" x to shape (32, 200) by repeating it 32 times along the first axis.
        # Since the gradient must be the same shape as the input, we must sum the gradient along the first axis to get the gradient of x in the backward pass if this was the case.
        # Same goes for y if x was shape (32, 200) and y was shape (200)
        if input_shape == grad_output.shape:
            return grad_output
        axes_to_sum = [
            i
            for i, (sx, sy) in enumerate(zip(input_shape, grad_output.shape))
            if sx != sy
        ]
        return grad_output.sum(axis=tuple(axes_to_sum), keepdims=True)



class Add(ElWiseFunction):
    def forward(self):
        self.ret = Tensor(
            self.x.data + self.y.data,
            requires_grad=self.requires_grad,
            storage=self.x.device,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        
        if self.x.requires_grad:
            grad = self._unbroadcast(grad_output, self.x.shape)
            self.x._grad += Tensor(grad, storage=self.storage).reshape(self.x.shape)
        if self.y.requires_grad:
            grad = self._unbroadcast(grad_output, self.y.shape)
            self.y._grad += Tensor(grad, storage=self.storage).reshape(self.y.shape)


class Mul(ElWiseFunction):
    def forward(self):
        self.ret = Tensor(
            self.x.data * self.y.data,
            requires_grad=self.requires_grad,
            storage=self.x.device,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        if self.x.requires_grad:
            grad = grad_output * self.y.data
            grad = self._unbroadcast(grad, self.x.shape)
            self.x._grad += Tensor(grad, storage=self.storage).reshape(self.x.shape)

        if self.y.requires_grad:
            grad = grad_output * self.x.data
            grad = self._unbroadcast(grad, self.y.shape)
            self.y._grad += Tensor(grad, storage=self.storage).reshape(self.y.shape)

class Div(ElWiseFunction):
    def forward(self):
        self.ret = Tensor(
            self.x.data / self.y.data,
            requires_grad=self.requires_grad,
            storage=self.x.device,
        )
        return self.ret

    def backward(self):
        grad_output = self.ret.grad.data
        if self.x.requires_grad:
            grad = grad_output / self.y.data
            grad = self._unbroadcast(grad, self.x.shape)
            self.x._grad += Tensor(grad, storage=self.storage).reshape(self.x.shape)

        if self.y.requires_grad:
            grad = -grad_output * self.x.data / (self.y.data ** 2)
            grad = self._unbroadcast(grad, self.y.shape)
            self.y._grad += Tensor(grad, storage=self.storage).reshape(self.y.shape)