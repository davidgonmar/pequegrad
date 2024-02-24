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
            and isinstance(condition, NumpyStorage)
            and isinstance(other, NumpyStorage)
            else NumpyStorage(np.where(condition.data, self.data, other))
        )

    @staticmethod
    def fill(shape: tuple, value: float) -> "NumpyStorage":
        return NumpyStorage(np.full(shape, value, dtype=np.float32))

    @staticmethod
    def where_static(
        condition: "NumpyStorage", x: "NumpyStorage", y: "NumpyStorage"
    ) -> "NumpyStorage":
        return (
            NumpyStorage(np.where(condition.data, x.data, y.data))
            if isinstance(y, NumpyStorage)
            and isinstance(x, NumpyStorage)
            and isinstance(condition, NumpyStorage)
            else NumpyStorage(
                np.where(
                    NumpyStorage(condition).data,
                    NumpyStorage(x).data,
                    NumpyStorage(y).data,
                )
            )
        )

    def im2col(self, kernel_shape: Tuple[int, int], stride: int = 1) -> "NumpyStorage":
        """
        Unfold a numpy array to a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
        It is equivalent to im2col transposed.

        Args:
            x: Input array of shape (batch_size, n_channels, x_h, x_w)
            kernel_shape: Kernel shape (k_h, k_w)
            stride: Stride (default: 1)

        Returns:
            Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))

        """
        x = self.data
        batch_size, in_channels, x_h, x_w = x.shape
        k_h, k_w = kernel_shape
        out_h = (x_h - k_h) // stride + 1
        out_w = (x_w - k_w) // stride + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid kernel size ({k_h}, {k_w}) or stride ({stride}) for input dimensions ({x_h}, {x_w})"
            )

        cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))

        for i in range(0, out_h):
            start_i = i * stride
            end_i = start_i + k_h
            for j in range(0, out_w):
                start_j = j * stride
                end_j = start_j + k_w
                cols[:, :, i * out_w + j] = x[
                    :, :, start_i:end_i, start_j:end_j
                ].reshape(batch_size, -1)

        return NumpyStorage(cols)

    def col2im(
        self,
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        stride: int = 1,
    ):
        """
        Fold a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
        It is equivalent to col2im transposed.

        Args:
            unfolded: Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
            kernel_shape: Kernel shape (k_h, k_w)
            output_shape: Output shape (x_h, x_w)
            stride: Stride (default: 1)

        Returns:
            Folded array of shape (batch_size, n_channels, x_h, x_w)

        """
        unfolded = self.data
        assert (
            len(unfolded.shape) == 3
        ), "unfolded must have 3 dimensions: (batch, k_h * k_w * n_channels, (out_h - k_h + 1) * (out_w - k_w + 1)), got shape {}".format(
            unfolded.shape
        )

        k_h, k_w = kernel_shape
        out_h, out_w = output_shape
        out_channels = unfolded.shape[1] // (k_h * k_w)
        out_batch = unfolded.shape[0]

        out = np.zeros((out_batch, out_channels, out_h, out_w))

        for i in range(0, (out_h - k_h) // stride + 1):
            for j in range(0, (out_w - k_w) // stride + 1):
                col = unfolded[:, :, i * ((out_w - k_w) // stride + 1) + j]
                start_i = i * stride
                end_i = start_i + k_h
                start_j = j * stride
                end_j = start_j + k_w
                out[:, :, start_i:end_i, start_j:end_j] += col.reshape(
                    out_batch, out_channels, k_h, k_w
                )

        return NumpyStorage(out)

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

    def __neg__(self) -> "NumpyStorage":
        return NumpyStorage(-self.data)
    
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

    def max(self, axis: int, keepdims: bool = False) -> "NumpyStorage":
        return NumpyStorage(np.max(self.data, axis, keepdims=keepdims))

    def mean(self, axis: int, keepdims: bool = False) -> "NumpyStorage":
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
