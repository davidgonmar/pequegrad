from typing import Set, Tuple, Union
from pequegrad.tensor import Tensor, CudaTensor, NumpyTensor

_Shape = Union[int, Tuple[int, ...]]

BackendTensor = Union[CudaTensor, NumpyTensor]


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
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.children = [t for t in tensors if t.requires_grad]

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
            Tensor(t, backend=device) if not isinstance(t, Tensor) else t
            for t in tensors
        ]
        # all devices should be the same
        assert all(
            t.backend == device for t in tensors
        ), "all tensors must be on the same device, got: {}".format(
            [t.device for t in tensors]
        )

        cls.backend = device

        f = cls(*tensors)
        ret = f.forward(*[t.data for t in tensors], **kwargs)
        ret = Tensor(ret, backend=device, requires_grad=f.requires_grad)
        ret._ctx = f if f.requires_grad else None

        assert (
            ret.device == device
        ), f"function output device {ret.device} does not match input device {device}"

        return ret
