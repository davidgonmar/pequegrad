import os
import sys
import warnings
from typing import Union
import numpy as np
from .utils import bind_method, bind_method_property


def make_cuda_tensor(
    data: Union[
        np.ndarray,
        "CudaTensor",
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
            data = CudaTensor.from_numpy(data)
        elif isinstance(data, (CudaTensor)):
            data = data
        elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
            data = CudaTensor.from_numpy(np.array(data))
        elif isinstance(data, CudaTensor):
            data = data.data
        else:
            raise ValueError(
                f"Data must be a numpy array or CudaTensor, got {data} of type {type(data)}"
            )
    else:
        if isinstance(data, CudaTensor) and data.dtype == dtype:
            data = data

        elif isinstance(data, (int, float, np.float32, np.float64, np.int32)):
            data = CudaTensor.from_numpy(np.array(data, dtype=dtype))

        elif isinstance(data, np.ndarray):
            if data.dtype != dtype:
                data = data.astype(
                    dtype if dtype in [np.float32, np.float64, np.int32] else np.float32
                )

            if data.dtype not in [np.float32, np.float64, np.int32]:
                warnings.warn(
                    f"Data type {data.dtype} is not supported, casting to float32"
                )

            data = CudaTensor.from_numpy(data)

    return data


class DummyCudaTensor:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise NotImplementedError("CUDA not available")

        return method

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CUDA not available")


SHOULD_USE_CUDA = (
    os.environ.get("PEQUEGRAD_USE_CUDA", "1") == "1"
)  # by default, use cuda

try:
    if SHOULD_USE_CUDA:
        build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
        if os.path.exists(build_path):
            sys.path.append(build_path)

        from pequegrad_cu import CudaTensor

        def mock_init(self, *args, **kwargs):
            pass

        bind_method(CudaTensor, "__getitem__", CudaTensor.slice)
        bind_method(CudaTensor, "__setitem__", CudaTensor.assign)
        bind_method(CudaTensor, "__neg__", lambda self: self.mul(-1))
        bind_method(CudaTensor, "numpy", CudaTensor.to_numpy)
        bind_method(CudaTensor, "new", make_cuda_tensor)
        bind_method(
            CudaTensor,
            "__new__",
            lambda cls, data, dtype=None: make_cuda_tensor(data, dtype),
        )
        bind_method(CudaTensor, "__init__", mock_init)
        bind_method(
            CudaTensor, "__repr__", lambda self: f"CudaTensor({self.to_numpy()})"
        )
        bind_method(CudaTensor, "power", CudaTensor.pow)
        bind_method(CudaTensor, "__matmul__", CudaTensor.matmul)
        bind_method(CudaTensor, "__len__", lambda self: self.shape[0])
        bind_method_property(CudaTensor, "size", lambda self: np.prod(self.shape))

        def T(self):
            axis = list(range(self.ndim))
            reversed_axis = axis[::-1]
            return self.permute(*reversed_axis)

        bind_method_property(CudaTensor, "T", T)
        CUDA_AVAILABLE = True

        CudaTensor = CudaTensor
    else:
        CudaTensor = DummyCudaTensor
        CUDA_AVAILABLE = False

except ImportError:
    CudaTensor = DummyCudaTensor
    CUDA_AVAILABLE = False
