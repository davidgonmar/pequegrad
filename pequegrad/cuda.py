import os
import sys


class DummyCudaArray:
    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise NotImplementedError("CUDA not available")

        return method


build_path = os.path.join(os.path.dirname(__file__), "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)

# todo - make it output somewhere else and make it general
windows_vs_build_path = os.path.join(
    os.path.dirname(__file__), "..", "out", "build", "x64-Debug"
)

if os.path.exists(windows_vs_build_path):
    sys.path.append(windows_vs_build_path)

try:
    from pequegrad_cu import Array as CudaArray  # noqa: F401, E402

    CUDA_AVAILABLE = True
except ImportError:
    CudaArray = DummyCudaArray
    CUDA_AVAILABLE = False
