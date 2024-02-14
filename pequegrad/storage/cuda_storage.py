import numpy as np
from typing import Union, Tuple
from pequegrad.cuda import CudaArray, CUDA_AVAILABLE
from .abstract_storage import AbstractStorage

if not CUDA_AVAILABLE:
    raise ImportError("CUDA is not available, still tried to import CudaStorage")


class CudaStorage(AbstractStorage):
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
    def T(self) -> "CudaStorage":
        axis = range(self.ndim)
        return self.permute(*reversed(axis))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def astype(self, dtype: Union[str, np.dtype]) -> "CudaStorage":
        raise NotImplementedError("Only float32 is supported")

    @property
    def strides(self) -> tuple:
        return tuple(self.data.strides)

    def __init__(
        self,
        data: Union[
            np.ndarray,
            CudaArray,
            int,
            float,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ],
    ):
        if isinstance(data, np.ndarray):
            self.data = CudaArray.from_numpy(
                data if data.dtype == np.float32 else data.astype(np.float32)
            )
        elif isinstance(data, CudaArray):
            self.data = data
        elif isinstance(data, (int, float, np.float32, np.float64, np.int32, np.int64)):
            self.data = CudaArray.from_numpy(np.array(data, dtype=np.float32))
        elif isinstance(data, CudaStorage):
            self.data = data.data
        else:
            raise ValueError(
                f"Data must be a numpy array or CudaArray, got {type(data)}"
            )

    def is_contiguous(self) -> bool:
        return self.data.is_contiguous()

    def add(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.add(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.add(other))
        )

    def __add__(self, other: "CudaStorage") -> "CudaStorage":
        return self.add(other)

    def __radd__(self, other: "CudaStorage") -> "CudaStorage":
        return self.add(other)

    def sub(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.sub(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.sub(other))
        )

    def __sub__(self, other: "CudaStorage") -> "CudaStorage":
        return self.sub(other)

    def __rsub__(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(other.data.sub(self.data))

    def mul(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.mul(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.mul(other))
        )

    def __mul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.mul(other)

    def __rmul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.mul(other)

    def matmul(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.matmul(other.data))

    def __matmul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.matmul(other)

    def div(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.div(other.data))

    def __truediv__(self, other: "CudaStorage") -> "CudaStorage":
        return self.div(other)

    def expand_dims(self, axis: int) -> "CudaStorage":
        return CudaStorage(self.data.unsqueeze(axis))

    def sum(self, axis: int = None, keepdims: bool = False) -> "CudaStorage":
        if axis is None:
            return CudaStorage(
                self.data.sum(keepdims)
            )  # todo - find a way not to have to do this if
        return CudaStorage(self.data.sum(axis, keepdims))

    def broadcast_to(self, shape: tuple) -> "CudaStorage":
        return CudaStorage(self.data.broadcast_to(shape))

    def contiguous(self) -> "CudaStorage":
        return CudaStorage(self.data.contiguous())

    def where(self, condition: "CudaStorage", other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.where(condition.data, other.data))
            if isinstance(other, CudaStorage) and isinstance(condition, CudaStorage)
            else CudaStorage(self.data.where(condition.data, other))
        )

    @staticmethod
    def where_static(
        condition: "CudaStorage", x: "CudaStorage", y: "CudaStorage"
    ) -> "CudaStorage":
        return (
            CudaStorage(x.data.where(condition.data, y.data))
            if isinstance(x, CudaStorage)
            and isinstance(y, CudaStorage)
            and isinstance(condition, CudaStorage)
            else CudaStorage(
                CudaStorage(x).data.where(
                    CudaStorage(condition).data, CudaStorage(y).data
                )
            )
        )

    def outer_product(self, other: "CudaStorage") -> "CudaStorage":
        raise NotImplementedError

    def swapaxes(self, axis1: int, axis2: int) -> "CudaStorage":
        a = list(range(self.ndim))
        a[axis1], a[axis2] = a[axis2], a[axis1]
        return self.permute(*a)

    def squeeze(self, axis: int) -> "CudaStorage":
        return CudaStorage(self.data.squeeze(axis))

    def equal(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.eq(other.data))

    def greater(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.gt(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.gt(CudaStorage(other).data))
        )

    def greater_equal(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.ge(other.data))

    def less(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.lt(other.data))

    def less_equal(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.le(other.data))

    def not_equal(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.ne(other.data))

    def __eq__(self, other: "CudaStorage") -> "CudaStorage":
        return self.equal(other)

    def __gt__(self, other: "CudaStorage") -> "CudaStorage":
        return self.greater(other)

    def __ge__(self, other: "CudaStorage") -> "CudaStorage":
        return self.greater_equal(other)

    def __lt__(self, other: "CudaStorage") -> "CudaStorage":
        return self.less(other)

    def __le__(self, other: "CudaStorage") -> "CudaStorage":
        return self.less_equal(other)

    def __ne__(self, other: "CudaStorage") -> "CudaStorage":
        return self.not_equal(other)

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "CudaStorage":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return CudaStorage(self.data.reshape(shape))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        return f"CudaStorage({self.data})"

    def max(self, axis: int, keepdims: bool = False) -> "CudaStorage":
        if axis is None:
            return CudaStorage(self.data.max(keepdims))
        return CudaStorage(self.data.max(axis, keepdims))

    def mean(
        self, axis: Union[int, None, Tuple] = None, keepdims: bool = False
    ) -> "CudaStorage":
        if axis is None:
            sum = self.data.sum(keepdims)
            N = np.prod(self.shape)
        else:
            sum = self.data.sum(axis, keepdims)
            if isinstance(axis, int):
                N = self.shape[axis]
            else:
                N = np.prod([self.shape[i] for i in axis])

        return CudaStorage(sum.div(CudaStorage(N).data))

    def pow(self, exponent: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.pow(exponent.data))
            if isinstance(exponent, CudaStorage)
            else CudaStorage(self.data.pow(CudaStorage(exponent).data))
        )

    def __pow__(self, exponent: "CudaStorage") -> "CudaStorage":
        return self.pow(exponent)

    def power(self, exponent: "CudaStorage") -> "CudaStorage":
        return self.pow(exponent)

    def log(self) -> "CudaStorage":
        return CudaStorage(self.data.log())

    def exp(self) -> "CudaStorage":
        return CudaStorage(self.data.exp())

    def permute(self, *dims: int) -> "CudaStorage":
        return CudaStorage(self.data.permute(dims))

    def el_wise_max(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.el_wise_max(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.el_wise_max(CudaStorage(other).data))
        )
