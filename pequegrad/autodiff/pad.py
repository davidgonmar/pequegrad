from pequegrad.tensor import _Shape
from pequegrad.autodiff.function import Function, BackendTensor


class PadConstant(Function):
    def forward(
        self, x: BackendTensor, pad: _Shape, constant: float = 0.0
    ) -> BackendTensor:
        pad = list(pad)  # for a 1d pad on last dim, it will be (padleft, padright)
        new_shape = list(x.shape)

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

        cls = x.__class__

        new_t = cls.fill(
            new_shape,
            constant,
            dtype=x.dtype,
        )

        slices = [slice(int(pad[0]), int(-pad[1])) for pad in padpairs]

        for i, _slice in enumerate(slices):
            if _slice.start == 0 and _slice.stop == 0:
                slices[i] = slice(None, None, None)  # same as a[:]

        slices = tuple(slices)
        new_t[slices] = x

        self.slices = slices
        return new_t

    def backward(self, grad_output: BackendTensor) -> BackendTensor:
        if self.requires_grad:
            return grad_output[self.slices]
