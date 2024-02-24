from typing import Set, Tuple, Union
from pequegrad.tensor import Tensor

_Shape = Union[int, Tuple[int, ...]]

class Function:
    ret: Tensor
    children: Set[Tensor]
    requires_grad: bool

    def __init__(self, *tensors: Tensor):
        assert all(
            isinstance(t, Tensor) for t in tensors
        ), "all inputs must be tensors, got: {}".format(
            list(filter(lambda t: not isinstance(t, Tensor), tensors))
        )
        self.requires_grad = any(t.requires_grad for t in tensors)
        self.children = set(t for t in tensors)

    def forward(self):
        raise NotImplementedError

    def backward(self) -> Tensor:
        raise NotImplementedError

    @classmethod
    def apply(
        cls,
        *tensors: Tensor,
        **kwargs,
    ) -> Tensor:
        # first, find first tensor that is a tensor
        device = "np"
        for t in tensors:
            if isinstance(t, Tensor):
                device = t.device
                break
        tensors = [
            Tensor(t, storage=device) if not isinstance(t, Tensor) else t
            for t in tensors
        ]
        # all devices should be the same
        assert all(
            t.storage_type == device for t in tensors
        ), "all tensors must be on the same device, got: {}".format(
            [t.device for t in tensors]
        )

        cls.storage = device

        f = cls(*tensors, **kwargs)
        f.forward()
        f.ret._ctx = f

        assert (
            f.ret.device == device
        ), f"function output device {f.ret.device} does not match input device {device}"

        return f.ret


