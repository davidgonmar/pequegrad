import numpy as np
from typing import Union, Tuple
from pequegrad.cuda import (
    CUDA_AVAILABLE,
    CudaArray,
)
import warnings
from .abstract_storage import AbstractStorage

if not CUDA_AVAILABLE:
    raise ImportError("CUDA is not available, still tried to import CudaStorage")


class CudaStorage(AbstractStorage):
    backend = "cuda"

    data: CudaArray

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype())

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
        return CudaStorage(self.data.astype(dtype))

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
        ],
        dtype: Union[str, np.dtype] = None,
    ):
        if dtype is None:
            if isinstance(data, np.ndarray):
                if data.dtype not in [np.float32, np.float64, np.int32]:
                    warnings.warn(
                        f"Data type {data.dtype} is not supported, casting to float32"
                    )
                    data = data.astype(np.float32)
                self.data = CudaArray.from_numpy(data)
            elif isinstance(data, (CudaArray)):
                self.data = data
            elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
                self.data = CudaArray.from_numpy(np.array(data))
            elif isinstance(data, CudaStorage):
                self.data = data.data
            else:
                raise ValueError(
                    f"Data must be a numpy array or CudaArray, got {type(data)}"
                )
        else:
            if isinstance(data, CudaArray) and data.dtype() == dtype:
                self.data = data

            elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
                self.data = CudaArray.from_numpy(np.array(data, dtype=dtype))

            elif isinstance(data, np.ndarray):
                if data.dtype != dtype:
                    data = data.astype(
                        dtype
                        if dtype in [np.float32, np.float64, np.int32]
                        else np.float32
                    )

                if data.dtype not in [np.float32, np.float64, np.int32]:
                    warnings.warn(
                        f"Data type {data.dtype} is not supported, casting to float32"
                    )

                self.data = CudaArray.from_numpy(data)

    def is_contiguous(self) -> bool:
        return self.data.is_contiguous()

    def add(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.add(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.add(other))
        )

    @staticmethod
    def fill(
        shape: Tuple[int, ...], value: Union[int, float], dtype=np.float64
    ) -> "CudaStorage":
        # transform value to dtype
        dtypestr = (
            "float64"
            if dtype == np.float64
            else "float32"
            if dtype == np.float32
            else "int32"
        )
        return CudaStorage(CudaArray.fill(shape, value, dtypestr))

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
        out = (
            CudaStorage(self.data.mul(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.mul(other))
        )

        return out

    def __mul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.mul(other)

    def __rmul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.mul(other)

    def matmul(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.matmul(other.data))

    def __matmul__(self, other: "CudaStorage") -> "CudaStorage":
        return self.matmul(other)

    def div(self, other: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.div(other.data))
            if isinstance(other, CudaStorage)
            else CudaStorage(self.data.div(other))
        )

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
        if not isinstance(condition, CudaStorage):
            condition = CudaStorage(condition, dtype=self.dtype)
        if not isinstance(other, CudaStorage):
            other = CudaStorage(other, dtype=self.dtype)

        return CudaStorage(self.data.where(condition.data, other.data))

    @staticmethod
    def where_static(
        condition: "CudaStorage", x: "CudaStorage", y: "CudaStorage"
    ) -> "CudaStorage":
        if not isinstance(condition, CudaStorage):
            condition = CudaStorage(condition, dtype=x.dtype)
        if not isinstance(x, CudaStorage):
            x = CudaStorage(x, dtype=condition.dtype)
        if not isinstance(y, CudaStorage):
            y = CudaStorage(y, dtype=condition.dtype)

        return CudaStorage(condition.data.where(x.data, y.data))

    def outer_product(self, other: "CudaStorage") -> "CudaStorage":
        return CudaStorage(self.data.outer_product(other.data))

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
            else CudaStorage(self.data.gt(CudaStorage(other, dtype=self.dtype).data))
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

    def __neg__(self) -> "CudaStorage":
        return CudaStorage(self.data.mul(-1))

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "CudaStorage":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return CudaStorage(self.data.reshape(list(shape)))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        return f"CudaStorage({self.data.to_numpy()})"

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

        return CudaStorage(sum.div(CudaStorage(N, dtype=self.dtype).data))

    def pow(self, exponent: "CudaStorage") -> "CudaStorage":
        return (
            CudaStorage(self.data.pow(exponent.data))
            if isinstance(exponent, CudaStorage)
            else CudaStorage(
                self.data.pow(CudaStorage(exponent, dtype=self.dtype).data)
            )
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
            else CudaStorage(
                self.data.el_wise_max(CudaStorage(other, dtype=self.dtype).data)
            )
        )

    def im2col(self, kernel_size: Tuple[int, int], stride: Union[int, Tuple[int, int]]):
        return CudaStorage(
            self.data.im2col(
                kernel_size, stride if isinstance(stride, Tuple) else (stride, stride)
            )
        )

    def col2im(
        self,
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        stride: Union[int, Tuple[int, int]],
    ) -> "CudaStorage":
        return CudaStorage(
            self.data.col2im(
                kernel_shape,
                output_shape,
                stride if isinstance(stride, Tuple) else (stride, stride),
            )
        )
