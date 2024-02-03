import numpy as np
from typing import Union, Tuple
from .cuda import CudaArray


class Storage:
    backend = "numpy"

    data: CudaArray

    @property
    def shape(self) -> tuple:
        return tuple(self.data.shape)

    @property
    def T(self) -> "Storage":
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError

    def astype(self, dtype: Union[str, np.dtype]) -> "Storage":
        raise NotImplementedError

    @property
    def strides(self) -> tuple:
        return tuple(self.data.strides)

    def __init__(self, data: np.ndarray):
        self.data = CudaArray.from_numpy(data)

    def add(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __add__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __radd__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def sub(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __sub__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __rsub__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def mul(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __mul__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __rmul__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def matmul(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __matmul__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def div(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __truediv__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def expand_dims(self, axis: int) -> "Storage":
        raise NotImplementedError

    def sum(self, axis: int = None, keepdims: bool = False) -> "Storage":
        raise NotImplementedError

    def broadcast_to(self, shape: tuple) -> "Storage":
        raise NotImplementedError

    def where(self, condition: "Storage", other: "Storage") -> "Storage":
        raise NotImplementedError

    def outer_product(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def swapaxes(self, axis1: int, axis2: int) -> "Storage":
        raise NotImplementedError

    def squeeze(self, axis: int) -> "Storage":
        raise NotImplementedError

    def equal(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def greater(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def greater_equal(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def less(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def less_equal(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __eq__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def __gt__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def __ge__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def __lt__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def __le__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def __ne__(self, other: "Storage") -> bool:
        raise NotImplementedError

    def numpy(self) -> np.ndarray:
        raise NotImplementedError

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Storage":
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def max(self, axis: int, keepdims: bool = None) -> "Storage":
        raise NotImplementedError

    def mean(self, axis: int, keepdims: bool = None) -> "Storage":
        raise NotImplementedError

    def pow(self, exponent: "Storage") -> "Storage":
        raise NotImplementedError

    def __pow__(self, exponent: "Storage") -> "Storage":
        raise NotImplementedError

    def power(self, exponent: "Storage") -> "Storage":
        raise NotImplementedError

    def log(self) -> "Storage":
        raise NotImplementedError

    def exp(self) -> "Storage":
        raise NotImplementedError

    def permute(self, *dims: int) -> "Storage":
        raise NotImplementedError

    def el_wise_max(self, other: "Storage") -> "Storage":
        raise NotImplementedError
