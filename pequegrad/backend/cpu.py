import os
import sys
import warnings
from typing import Union
import numpy as np


class DummyCpuTensor:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise NotImplementedError("CPU C++ library not available")

        return method

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CPU C++ library not available")


def make_cuda_tensor(
    data: Union[
        np.ndarray,
        "CpuTensor",
        int,
        float,
        list,
        np.float32,
        np.float64,
        np.int32,
    ],
    dtype: Union[str, np.dtype] = None,
):
    if isinstance(data, list):
        data = np.array(data)
    if dtype is None:
        if isinstance(data, np.ndarray):
            if data.dtype not in [np.float32, np.float64, np.int32]:
                warnings.warn(
                    f"Data type {data.dtype} is not supported, casting to float32"
                )
                data = data.astype(np.float32)
            data = CpuTensor.from_numpy(data)
        elif isinstance(data, (CpuTensor)):
            data = data
        elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
            data = CpuTensor.from_numpy(np.array(data))
        elif isinstance(data, CpuTensor):
            data = data.data
        else:
            raise ValueError(
                f"Data must be a numpy array or CpuTensor, got {data} of type {type(data)}"
            )
    else:
        if isinstance(data, CpuTensor) and data.dtype == dtype:
            data = data

        elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
            data = CpuTensor.from_numpy(np.array(data, dtype=dtype))

        elif isinstance(data, np.ndarray):
            if data.dtype != dtype:
                data = data.astype(
                    dtype if dtype in [np.float32, np.float64, np.int32] else np.float32
                )

            if data.dtype not in [np.float32, np.float64, np.int32]:
                warnings.warn(
                    f"Data type {data.dtype} is not supported, casting to float32"
                )

            data = CpuTensor.from_numpy(data)

    return data


try:
    build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
    if os.path.exists(build_path):
        sys.path.append(build_path)

    from pequegrad_cpu import CpuTensor  # noqa

    from .utils import bind_method

    bind_method(CpuTensor, "__repr__", lambda self: f"CpuTensor({self.to_numpy()})")
    bind_method(CpuTensor, "__add__", lambda self, other: self.add(other))
    bind_method(
        CpuTensor,
        "__new__",
        lambda cls, data, dtype=None: make_cuda_tensor(data, dtype),
    )
    bind_method(CpuTensor, "__init__", lambda *args, **kwargs: None)
    bind_method(CpuTensor, "numpy", lambda self: self.to_numpy())
    CPU_AVAILABLE = True
except ImportError as e:
    print(e)
    CpuTensor = DummyCpuTensor
    CPU_AVAILABLE = False
