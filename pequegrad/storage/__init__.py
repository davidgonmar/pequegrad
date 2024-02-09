from .abstract_storage import AbstractStorage  # noqa

from .numpy_storage import NumpyStorage  # noqa
from pequegrad.cuda import CUDA_AVAILABLE  # noqa

if CUDA_AVAILABLE:
    from .cuda_storage import CudaStorage  # noqaq
else:
    from .cuda_storage_dummy import DummyCudaStorage as CudaStorage  # noqa
