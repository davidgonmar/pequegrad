from pequegrad.tensor import Tensor, _Shape
from pequegrad.autodiff.function import Function


class PadConstant(Function):
    def __init__(self, x: Tensor, pad: _Shape, constant: int = 0):
        super().__init__(x)
        self.x = x
        self.pad = pad
        self.constant = constant

    def forward(self):
        pad = list(self.pad)  # for a 1d pad on last dim, it will be (padleft, padright)
        new_shape = list(self.x.shape)

        padpairs = list(
            zip(pad[::2], pad[1::2])
        )  # will split (a, b, c, d) into [(a, b), (c, d)]
        padpairs = list(reversed(padpairs))  # to match torch's behavior
        # which means "on last dim, pad a on left, b on right, and on last-1 dim, pad c on left, d on right"

        # pad padpairs with 0 for each dimension that is not being padded, to the start of the list
        for _ in range(len(new_shape) - len(padpairs)):
            padpairs.insert(0, (0, 0))

        # now we can calculate the new shape
        for i, (padleft, padright) in enumerate(padpairs):
            new_shape[i] += padleft + padright

        new_t = Tensor.fill(
            new_shape,
            self.constant,
            requires_grad=self.requires_grad,
            backend=self.backend,
        )

        slices = [slice(int(pad[0]), int(-pad[1])) for pad in padpairs]

        for i, _slice in enumerate(slices):
            if _slice.start == 0 and _slice.stop == 0:
                slices[i] = slice(None, None, None) # same as a[:]

        slices = tuple(slices)
        self.slices = slices  # save for backward
        new_t[slices] = self.x.data

        self.ret = new_t
        return self.ret

    def backward(self):
        if self.x.requires_grad:
            self.x._grad += self.ret.grad[self.slices]
