from .function import Function, BackendTensor


class ElWiseFunction(Function):
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
        x = grad_output.sum(axis=tuple(axes_to_sum), keepdims=True)

        return x


class Add(ElWiseFunction):
    def forward(self, x: BackendTensor, y: BackendTensor):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x + y

    def backward(self, grad_output: BackendTensor):
        x_grad, y_grad = None, None
        if self.needs_input_grad[0]:
            x_grad = self._unbroadcast(grad_output, self.x_shape).reshape(*self.x_shape)
        if self.needs_input_grad[1]:
            y_grad = self._unbroadcast(grad_output, self.y_shape).reshape(*self.y_shape)
        return x_grad, y_grad


class Mul(ElWiseFunction):
    def forward(self, x: BackendTensor, y: BackendTensor):
        self.x, self.y = (x, y) if self.requires_grad else (None, None)
        return x * y

    def backward(self, grad_output: BackendTensor):
        x_grad, y_grad = None, None
        if self.needs_input_grad[0]:
            x_grad = self._unbroadcast(grad_output * self.y, self.x.shape).reshape(
                *self.x.shape
            )
        if self.needs_input_grad[1]:
            y_grad = self._unbroadcast(grad_output * self.x, self.y.shape).reshape(
                *self.y.shape
            )
        return x_grad, y_grad


class Div(ElWiseFunction):
    def forward(self, x: BackendTensor, y: BackendTensor):
        self.x, self.y = (
            (x, y) if self.requires_grad else (None, None)
        )  # save for backward pass both
        return x / y

    def backward(self, grad_output: BackendTensor):
        x_grad, y_grad = None, None
        if self.needs_input_grad[0]:
            grad = grad_output / self.y
            grad = self._unbroadcast(grad, self.x.shape)
            x_grad = grad.reshape(*self.x.shape)

        if self.needs_input_grad[1]:
            grad = -grad_output * self.x / (self.y**2)
            grad = self._unbroadcast(grad, self.y.shape)
            y_grad = grad.reshape(*self.y.shape)

        return x_grad, y_grad
