import numpy as np
from typing import Union, Tuple
from .cuda import CudaArray


class Storage:
    backend = "cuda"

    data: CudaArray

    def to_numpy(self) -> np.ndarray:
        return self.data.to_numpy()

    def numpy(self) -> np.ndarray:
        return self.to_numpy()

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
        raise NotImplementedError("Only float32 is supported")

    @property
    def strides(self) -> tuple:
        return tuple(self.data.strides)

    def __init__(self, data: np.ndarray):
        if isinstance(data, np.ndarray):
            self.data = CudaArray.from_numpy(data)
        elif isinstance(data, CudaArray):
            self.data = data.clone()
        else:
            raise ValueError(
                f"Data must be a numpy array or CudaArray, got {type(data)}"
            )

    def add(self, other: "Storage") -> "Storage":
        return Storage(self.data.add(other.data))

    def __add__(self, other: "Storage") -> "Storage":
        return self.add(other)

    def __radd__(self, other: "Storage") -> "Storage":
        raise self.add(other)

    def sub(self, other: "Storage") -> "Storage":
        return Storage(self.data.sub(other.data))

    def __sub__(self, other: "Storage") -> "Storage":
        return self.sub(other)

    def __rsub__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def mul(self, other: "Storage") -> "Storage":
        return Storage(self.data.mul(other.data))

    def __mul__(self, other: "Storage") -> "Storage":
        return self.mul(other)

    def __rmul__(self, other: "Storage") -> "Storage":
        return self.mul(other)

    def matmul(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def __matmul__(self, other: "Storage") -> "Storage":
        raise NotImplementedError

    def div(self, other: "Storage") -> "Storage":
        return Storage(self.data.div(other.data))

    def __truediv__(self, other: "Storage") -> "Storage":
        return self.div(other)

    def expand_dims(self, axis: int) -> "Storage":
        raise NotImplementedError

    def sum(self, axis: int = None, keepdims: bool = False) -> "Storage":
        raise NotImplementedError

    def broadcast_to(self, shape: tuple) -> "Storage":
        return Storage(self.data.broadcast_to(shape))

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

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Storage":
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        return f"Storage({self.data})"

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