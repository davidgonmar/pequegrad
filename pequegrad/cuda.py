import os
import sys
import warnings


class DummyCudaArray:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise NotImplementedError("CUDA not available")

        return method


SHOULD_USE_CUDA = (
    os.environ.get("PEQUEGRAD_USE_CUDA", "1") == "1"
)  # by default, use cuda

try:
    if SHOULD_USE_CUDA:
        build_path = os.path.join(os.path.dirname(__file__), "..", "build")
        if os.path.exists(build_path):
            sys.path.append(build_path)

        from pequegrad_cu import Array as CudaArray

        CUDA_AVAILABLE = True
    else:
        CudaArray = DummyCudaArray
        CUDA_AVAILABLE = False

except ImportError:
    warnings.warn("Tried to use cuda, but it is not available")
    CudaArray = DummyCudaArray
    CUDA_AVAILABLE = False
