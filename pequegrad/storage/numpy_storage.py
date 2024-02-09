import numpy as np
from typing import Union, Tuple
from .abstract_storage import AbstractStorage


class NumpyStorage(AbstractStorage):
    backend = "numpy"

    data: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> "NumpyStorage":
        return NumpyStorage(self.data.T)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def astype(self, dtype: Union[str, np.dtype]) -> "NumpyStorage":
        return NumpyStorage(self.data.astype(dtype))

    @property
    def strides(self) -> tuple:
        return self.data.strides

    def __init__(self, data: np.ndarray):
        if isinstance(data, NumpyStorage):
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

    def add(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(self.data + other.data)
            if isinstance(other, NumpyStorage)
            else NumpyStorage(self.data + other)
        )

    def __add__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.add(other)

    def __radd__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.add(other)

    def sub(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(self.data - other.data)
            if isinstance(other, NumpyStorage)
            else NumpyStorage(self.data - other)
        )

    def __sub__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.sub(other)

    def __rsub__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.sub(other)

    def mul(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(self.data * other.data)
            if isinstance(other, NumpyStorage)
            else NumpyStorage(self.data * other)
        )

    def __mul__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.mul(other)

    def __rmul__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.mul(other)

    def matmul(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(self.data @ other.data)
            if isinstance(other, NumpyStorage)
            else NumpyStorage(self.data @ other)
        )

    def __matmul__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.matmul(other)

    def div(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(self.data / other.data)
            if isinstance(other, NumpyStorage)
            else NumpyStorage(self.data / other)
        )

    def __truediv__(self, other: "NumpyStorage") -> "NumpyStorage":
        return self.div(other)

    def expand_dims(self, axis: int) -> "NumpyStorage":
        return NumpyStorage(np.expand_dims(self.data, axis))

    def sum(self, axis: int = None, keepdims: bool = False) -> "NumpyStorage":
        return NumpyStorage(np.sum(self.data, axis, keepdims=keepdims))

    def broadcast_to(self, shape: tuple) -> "NumpyStorage":
        return NumpyStorage(np.broadcast_to(self.data, shape))

    def where(self, condition: "NumpyStorage", other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.where(condition.data, self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.where(condition.data, self.data, other))
        )

    def outer_product(self, other: "NumpyStorage") -> "NumpyStorage":
        return NumpyStorage(np.outer(self.data, other.data))

    def swapaxes(self, axis1: int, axis2: int) -> "NumpyStorage":
        return NumpyStorage(np.swapaxes(self.data, axis1, axis2))

    def contiguous(self) -> "NumpyStorage":
        return NumpyStorage(self.data.copy())

    def is_contiguous(self) -> bool:
        return self.data.flags["C_CONTIGUOUS"]

    def squeeze(self, axis: int) -> "NumpyStorage":
        return NumpyStorage(np.squeeze(self.data, axis))

    def equal(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.equal(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.equal(self.data, other))
        )

    def greater(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.greater(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.greater(self.data, other))
        )

    def greater_equal(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.greater_equal(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.greater_equal(self.data, other))
        )

    def less(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.less(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.less(self.data, other))
        )

    def less_equal(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.less_equal(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.less_equal(self.data, other))
        )

    def not_equal(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.not_equal(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.not_equal(self.data, other))
        )

    def __eq__(self, other: "NumpyStorage") -> bool:
        return self.equal(other)

    def __gt__(self, other: "NumpyStorage") -> bool:
        return self.greater(other)

    def __ge__(self, other: "NumpyStorage") -> bool:
        return self.greater_equal(other)

    def __lt__(self, other: "NumpyStorage") -> bool:
        return self.less(other)

    def __le__(self, other: "NumpyStorage") -> bool:
        return self.less_equal(other)

    def __ne__(self, other: "NumpyStorage") -> bool:
        return self.not_equal(other)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "NumpyStorage":
        # Flatten the shape input to handle a single tuple or multiple int arguments
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return NumpyStorage(np.reshape(self.data, shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return NumpyStorage(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data

    def __repr__(self):
        assert type(self.data) == np.ndarray
        return f"NumpyStorage(npdata={self.data})"

    def max(self, axis: int, keepdims: bool = None) -> "NumpyStorage":
        return NumpyStorage(np.max(self.data, axis, keepdims=keepdims))

    def mean(self, axis: int, keepdims: bool = None) -> "NumpyStorage":
        return NumpyStorage(np.mean(self.data, axis, keepdims=keepdims))

    def pow(self, exponent: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.power(self.data, exponent.data))
            if isinstance(exponent, NumpyStorage)
            else NumpyStorage(np.power(self.data, exponent))
        )

    def __pow__(self, exponent: "NumpyStorage") -> "NumpyStorage":
        return self.pow(exponent)

    def power(self, exponent: "NumpyStorage") -> "NumpyStorage":
        return self.pow(exponent)

    def log(self) -> "NumpyStorage":
        return NumpyStorage(np.log(self.data))

    def exp(self) -> "NumpyStorage":
        return NumpyStorage(np.exp(self.data))

    def permute(self, *dims: int) -> "NumpyStorage":
        return NumpyStorage(np.transpose(self.data, dims))

    def el_wise_max(self, other: "NumpyStorage") -> "NumpyStorage":
        return (
            NumpyStorage(np.maximum(self.data, other.data))
            if isinstance(other, NumpyStorage)
            else NumpyStorage(np.maximum(self.data, other))
        )
