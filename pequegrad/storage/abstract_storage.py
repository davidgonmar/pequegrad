from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple, Literal


class AbstractStorage(ABC):
    data: any
    backend: Literal["np", "cuda"]

    @abstractmethod
    def __init__(self, data: np.ndarray):
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def T(self) -> "AbstractStorage":
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def astype(self, dtype: Union[str, np.dtype]) -> "AbstractStorage":
        pass

    @property
    @abstractmethod
    def strides(self) -> tuple:
        pass

    @abstractmethod
    def add(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def sub(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def mul(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def matmul(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def div(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def expand_dims(self, axis: int) -> "AbstractStorage":
        pass

    @abstractmethod
    def sum(self, axis: int = None, keepdims: bool = False) -> "AbstractStorage":
        pass

    @abstractmethod
    def broadcast_to(self, shape: tuple) -> "AbstractStorage":
        pass

    @abstractmethod
    def where(
        self, condition: "AbstractStorage", other: "AbstractStorage"
    ) -> "AbstractStorage":
        pass

    @abstractmethod
    def outer_product(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def swapaxes(self, axis1: int, axis2: int) -> "AbstractStorage":
        pass

    @abstractmethod
    def contiguous(self) -> "AbstractStorage":
        pass

    @abstractmethod
    def is_contiguous(self) -> bool:
        pass

    @abstractmethod
    def squeeze(self, axis: int) -> "AbstractStorage":
        pass

    @abstractmethod
    def equal(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def greater(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def greater_equal(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def less(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def less_equal(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def not_equal(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "AbstractStorage":
        pass

    @abstractmethod
    def max(self, axis: int, keepdims: bool = None) -> "AbstractStorage":
        pass

    @abstractmethod
    def mean(self, axis: int, keepdims: bool = None) -> "AbstractStorage":
        pass

    @abstractmethod
    def pow(self, exponent: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def log(self) -> "AbstractStorage":
        pass

    @abstractmethod
    def exp(self) -> "AbstractStorage":
        pass

    @abstractmethod
    def permute(self, *dims: int) -> "AbstractStorage":
        pass

    @abstractmethod
    def el_wise_max(self, other: "AbstractStorage") -> "AbstractStorage":
        pass

    @abstractmethod
    def im2col(self, kernel_size: int, stride: int) -> "AbstractStorage":
        pass

    @abstractmethod
    def col2im(self, input_shape: tuple, kernel_size: int, stride: int) -> "AbstractStorage":
        pass