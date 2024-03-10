import numpy as np
from typing import Union, Tuple


class NumpyTensor:
    backend = "numpy"

    data: np.ndarray

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def T(self) -> "NumpyTensor":
        return NumpyTensor(self.data.T)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def astype(self, dtype: Union[str, np.dtype]) -> "NumpyTensor":
        return NumpyTensor(self.data.astype(dtype))

    @property
    def strides(self) -> tuple:
        return self.data.strides

    def __init__(self, data: np.ndarray):
        if isinstance(data, NumpyTensor):
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

        assert (
            data.dtype != object
        ), "Data type not supported, first element is {}".format(data[0])
        self.data = data

    def add(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(self.data + other.data)
            if isinstance(other, NumpyTensor)
            else NumpyTensor(self.data + other)
        )

    def __add__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.add(other)

    def __radd__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.add(other)

    def sub(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(self.data - other.data)
            if isinstance(other, NumpyTensor)
            else NumpyTensor(self.data - other)
        )

    def __sub__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.sub(other)

    def __rsub__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.sub(other)

    def mul(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(self.data * other.data)
            if isinstance(other, NumpyTensor)
            else NumpyTensor(self.data * other)
        )

    def __mul__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.mul(other)

    def __rmul__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.mul(other)

    def matmul(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(self.data @ other.data)
            if isinstance(other, NumpyTensor)
            else NumpyTensor(self.data @ other)
        )

    def __matmul__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.matmul(other)

    def div(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(self.data / other.data)
            if isinstance(other, NumpyTensor)
            else NumpyTensor(self.data / other)
        )

    def __truediv__(self, other: "NumpyTensor") -> "NumpyTensor":
        return self.div(other)

    def expand_dims(self, axis: int) -> "NumpyTensor":
        return NumpyTensor(np.expand_dims(self.data, axis))

    def sum(self, axis: int = None, keepdims: bool = False) -> "NumpyTensor":
        return NumpyTensor(np.sum(self.data, axis, keepdims=keepdims))

    def broadcast_to(self, shape: tuple) -> "NumpyTensor":
        return NumpyTensor(np.broadcast_to(self.data, shape))

    def where(self, a: "NumpyTensor", b: "NumpyTensor") -> "NumpyTensor":
        if not isinstance(a, NumpyTensor):
            a = NumpyTensor(a)
        if not isinstance(b, NumpyTensor):
            b = NumpyTensor(b)
        return NumpyTensor(np.where(self.data, a.data, b.data))

    @staticmethod
    def fill(
        shape: tuple, value: Union[int, float], dtype: np.dtype = np.float32
    ) -> "NumpyTensor":
        return NumpyTensor(np.full(shape, value, dtype=dtype))

    @staticmethod
    def where_static(
        condition: "NumpyTensor", x: "NumpyTensor", y: "NumpyTensor"
    ) -> "NumpyTensor":
        return (
            NumpyTensor(np.where(condition.data, x.data, y.data))
            if isinstance(y, NumpyTensor)
            and isinstance(x, NumpyTensor)
            and isinstance(condition, NumpyTensor)
            else NumpyTensor(
                np.where(
                    NumpyTensor(condition).data,
                    NumpyTensor(x).data,
                    NumpyTensor(y).data,
                )
            )
        )

    def im2col(
        self,
        kernel_shape: Tuple[int, int],
        stride: Union[Tuple[int, int], int] = 1,
        dilation: Union[Tuple[int, int], int] = 1,
    ) -> "NumpyTensor":
        """
        Unfold a numpy array to a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
        It is equivalent to im2col transposed.

        Args:
            x: Input array of shape (batch_size, n_channels, x_h, x_w)
            kernel_shape: Kernel shape (k_h, k_w)
            stride: stride (s_h, s_w)

        Returns:
            Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))

        """
        x = self.data
        batch_size, in_channels, x_h, x_w = x.shape
        k_h, k_w = kernel_shape
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        assert d_h > 0 and d_w > 0, "Dilation must be greater than 0"
        out_h = (x_h - (d_h * (k_h - 1)) - 1) // s_h + 1
        out_w = (x_w - (d_w * (k_w - 1)) - 1) // s_w + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid kernel size ({k_h}, {k_w}) or strides ({s_h}, {s_w}) for input size ({x_h}, {x_w})"
            )

        cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))

        for i in range(0, out_h):
            start_i = i * s_h
            end_i = start_i + (k_h - 1) * d_h + 1
            for j in range(0, out_w):
                start_j = j * s_w
                end_j = start_j + (k_w - 1) * d_w + 1
                cols[:, :, i * out_w + j] = x[
                    :, :, start_i:end_i:d_h, start_j:end_j:d_w
                ].reshape(batch_size, -1)

        return NumpyTensor(cols)

    def col2im(
        self,
        kernel_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        stride: Union[Tuple[int, int], int] = 1,
        dilation: Union[Tuple[int, int], int] = 1,
    ):
        """
        Fold a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
        It is equivalent to col2im transposed.

        Args:
            unfolded: Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
            kernel_shape: Kernel shape (k_h, k_w)
            output_shape: Output shape (x_h, x_w)
            stride: stride (s_h, s_w)

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
        s_h, s_w = stride if isinstance(stride, tuple) else (stride, stride)
        d_h, d_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        out_h, out_w = output_shape
        out_channels = unfolded.shape[1] // (k_h * k_w)
        out_batch = unfolded.shape[0]

        out = np.zeros((out_batch, out_channels, out_h, out_w))

        for i in range(0, (out_h - (k_h - 1) * d_w - 1) // s_h + 1):
            for j in range(0, (out_w - (k_w - 1) * d_w - 1) // s_w + 1):
                col = unfolded[:, :, i * ((out_w - (k_w - 1) * d_w - 1) // s_w + 1) + j]
                start_i = i * s_h
                end_i = start_i + (k_h - 1) * d_h + 1
                start_j = j * s_w
                end_j = start_j + (k_w - 1) * d_w + 1
                out[:, :, start_i:end_i:d_h, start_j:end_j:d_w] += col.reshape(
                    out_batch, out_channels, k_h, k_w
                )

        return NumpyTensor(out)

    def outer_product(self, other: "NumpyTensor") -> "NumpyTensor":
        return NumpyTensor(np.outer(self.data, other.data))

    def swapaxes(self, axis1: int, axis2: int) -> "NumpyTensor":
        return NumpyTensor(np.swapaxes(self.data, axis1, axis2))

    def contiguous(self) -> "NumpyTensor":
        return NumpyTensor(self.data.copy())

    def is_contiguous(self) -> bool:
        return self.data.flags["C_CONTIGUOUS"]

    def squeeze(self, axis: int) -> "NumpyTensor":
        return NumpyTensor(np.squeeze(self.data, axis))

    def equal(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.equal(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.equal(self.data, other))
        )

    def greater(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.greater(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.greater(self.data, other))
        )

    def greater_equal(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.greater_equal(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.greater_equal(self.data, other))
        )

    def less(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.less(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.less(self.data, other))
        )

    def less_equal(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.less_equal(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.less_equal(self.data, other))
        )

    def not_equal(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.not_equal(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.not_equal(self.data, other))
        )

    def __eq__(self, other: "NumpyTensor") -> bool:
        return self.equal(other)

    def __gt__(self, other: "NumpyTensor") -> bool:
        return self.greater(other)

    def __ge__(self, other: "NumpyTensor") -> bool:
        return self.greater_equal(other)

    def __lt__(self, other: "NumpyTensor") -> bool:
        return self.less(other)

    def __le__(self, other: "NumpyTensor") -> bool:
        return self.less_equal(other)

    def __neg__(self) -> "NumpyTensor":
        return NumpyTensor(-self.data)

    def __ne__(self, other: "NumpyTensor") -> bool:
        return self.not_equal(other)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "NumpyTensor":
        # Flatten the shape input to handle a single tuple or multiple int arguments
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return NumpyTensor(np.reshape(self.data, shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return NumpyTensor(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = (
            value
            if isinstance(value, np.ndarray)
            else value.data
            if isinstance(value, NumpyTensor)
            else value
        )

    def __repr__(self):
        assert type(self.data) == np.ndarray
        return f"NumpyTensor(npdata={self.data})"

    def max(self, axis: int, keepdims: bool = False) -> "NumpyTensor":
        return NumpyTensor(np.max(self.data, axis, keepdims=keepdims))

    def mean(self, axis: int, keepdims: bool = False) -> "NumpyTensor":
        return NumpyTensor(
            np.mean(self.data, axis, keepdims=keepdims, dtype=self.data.dtype)
        )

    def pow(self, exponent: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.power(self.data, exponent.data))
            if isinstance(exponent, NumpyTensor)
            else NumpyTensor(np.power(self.data, exponent))
        )

    def __pow__(self, exponent: "NumpyTensor") -> "NumpyTensor":
        return self.pow(exponent)

    def power(self, exponent: "NumpyTensor") -> "NumpyTensor":
        return self.pow(exponent)

    def log(self) -> "NumpyTensor":
        return NumpyTensor(np.log(self.data))

    def exp(self) -> "NumpyTensor":
        return NumpyTensor(np.exp(self.data))

    def permute(self, *dims: int) -> "NumpyTensor":
        return NumpyTensor(np.transpose(self.data, dims))

    def el_wise_max(self, other: "NumpyTensor") -> "NumpyTensor":
        return (
            NumpyTensor(np.maximum(self.data, other.data))
            if isinstance(other, NumpyTensor)
            else NumpyTensor(np.maximum(self.data, other))
        )
