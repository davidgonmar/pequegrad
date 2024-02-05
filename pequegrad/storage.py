import numpy as np
from typing import Union, Tuple


class Storage:
    backend = "numpy"

    data: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> "Storage":
        return Storage(self.data.T)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def astype(self, dtype: Union[str, np.dtype]) -> "Storage":
        return Storage(self.data.astype(dtype))

    @property
    def strides(self) -> tuple:
        return self.data.strides

    def __init__(self, data: np.ndarray):
        if isinstance(data, Storage):
            data = data.data
        # scalars
        if isinstance(
            data, (int, float, np.float32, np.float64, np.int32, np.int64, np.bool_)
        ):
            # if bool, convert to int
            if isinstance(data, np.bool_):
                data = int(data)
            data = np.array(data)

        assert isinstance(
            data, np.ndarray
        ), f"Data must be a numpy array, got {type(data)}"

        assert data.dtype != object, "Data type not supported"
        self.data = data

    def add(self, other: "Storage") -> "Storage":
        return (
            Storage(self.data + other.data)
            if isinstance(other, Storage)
            else Storage(self.data + other)
        )

    def __add__(self, other: "Storage") -> "Storage":
        return self.add(other)

    def __radd__(self, other: "Storage") -> "Storage":
        return self.add(other)

    def sub(self, other: "Storage") -> "Storage":
        return (
            Storage(self.data - other.data)
            if isinstance(other, Storage)
            else Storage(self.data - other)
        )

    def __sub__(self, other: "Storage") -> "Storage":
        return self.sub(other)

    def __rsub__(self, other: "Storage") -> "Storage":
        return self.sub(other)

    def mul(self, other: "Storage") -> "Storage":
        return (
            Storage(self.data * other.data)
            if isinstance(other, Storage)
            else Storage(self.data * other)
        )

    def __mul__(self, other: "Storage") -> "Storage":
        return self.mul(other)

    def __rmul__(self, other: "Storage") -> "Storage":
        return self.mul(other)

    def matmul(self, other: "Storage") -> "Storage":
        return (
            Storage(self.data @ other.data)
            if isinstance(other, Storage)
            else Storage(self.data @ other)
        )

    def __matmul__(self, other: "Storage") -> "Storage":
        return self.matmul(other)

    def div(self, other: "Storage") -> "Storage":
        return (
            Storage(self.data / other.data)
            if isinstance(other, Storage)
            else Storage(self.data / other)
        )

    def __truediv__(self, other: "Storage") -> "Storage":
        return self.div(other)

    def expand_dims(self, axis: int) -> "Storage":
        return Storage(np.expand_dims(self.data, axis))

    def sum(self, axis: int = None, keepdims: bool = False) -> "Storage":
        return Storage(np.sum(self.data, axis, keepdims=keepdims))

    def broadcast_to(self, shape: tuple) -> "Storage":
        return Storage(np.broadcast_to(self.data, shape))

    def where(self, condition: "Storage", other: "Storage") -> "Storage":
        return (
            Storage(np.where(condition.data, self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.where(condition.data, self.data, other))
        )

    def outer_product(self, other: "Storage") -> "Storage":
        return Storage(np.outer(self.data, other.data))

    def swapaxes(self, axis1: int, axis2: int) -> "Storage":
        return Storage(np.swapaxes(self.data, axis1, axis2))

    def contiguous(self) -> "Storage":
        return Storage(self.data.copy())

    def squeeze(self, axis: int) -> "Storage":
        return Storage(np.squeeze(self.data, axis))

    def equal(self, other: "Storage") -> "Storage":
        return (
            Storage(np.equal(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.equal(self.data, other))
        )

    def greater(self, other: "Storage") -> "Storage":
        return (
            Storage(np.greater(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.greater(self.data, other))
        )

    def greater_equal(self, other: "Storage") -> "Storage":
        return (
            Storage(np.greater_equal(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.greater_equal(self.data, other))
        )

    def less(self, other: "Storage") -> "Storage":
        return (
            Storage(np.less(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.less(self.data, other))
        )

    def less_equal(self, other: "Storage") -> "Storage":
        return (
            Storage(np.less_equal(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.less_equal(self.data, other))
        )

    def not_equal(self, other: "Storage") -> "Storage":
        return (
            Storage(np.not_equal(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.not_equal(self.data, other))
        )

    def __eq__(self, other: "Storage") -> bool:
        return self.equal(other)

    def __gt__(self, other: "Storage") -> bool:
        return self.greater(other)

    def __ge__(self, other: "Storage") -> bool:
        return self.greater_equal(other)

    def __lt__(self, other: "Storage") -> bool:
        return self.less(other)

    def __le__(self, other: "Storage") -> bool:
        return self.less_equal(other)

    def __ne__(self, other: "Storage") -> bool:
        return self.not_equal(other)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Storage":
        # Flatten the shape input to handle a single tuple or multiple int arguments
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Storage(np.reshape(self.data, shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return Storage(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data

    def __repr__(self):
        assert type(self.data) == np.ndarray
        return f"Storage(npdata={self.data})"

    def max(self, axis: int, keepdims: bool = None) -> "Storage":
        return Storage(np.max(self.data, axis, keepdims=keepdims))

    def mean(self, axis: int, keepdims: bool = None) -> "Storage":
        return Storage(np.mean(self.data, axis, keepdims=keepdims))

    def pow(self, exponent: "Storage") -> "Storage":
        return (
            Storage(np.power(self.data, exponent.data))
            if isinstance(exponent, Storage)
            else Storage(np.power(self.data, exponent))
        )

    def __pow__(self, exponent: "Storage") -> "Storage":
        return self.pow(exponent)

    def power(self, exponent: "Storage") -> "Storage":
        return self.pow(exponent)

    def log(self) -> "Storage":
        return Storage(np.log(self.data))

    def exp(self) -> "Storage":
        return Storage(np.exp(self.data))

    def permute(self, *dims: int) -> "Storage":
        return Storage(np.transpose(self.data, dims))

    def el_wise_max(self, other: "Storage") -> "Storage":
        return (
            Storage(np.maximum(self.data, other.data))
            if isinstance(other, Storage)
            else Storage(np.maximum(self.data, other))
        )
