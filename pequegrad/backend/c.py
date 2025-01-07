import os
import sys

# set stack size to 64MB

sys.setrecursionlimit(10**6)
# todo -- it cannot find the CUDA dlls for nvrtc if I don't add the path here
paths = [
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin",
    # cudnn
    "C:/Program Files/NVIDIA/CUDNN/v9.2/bin/12.5",
]


for path in paths:
    if os.path.exists(path):
        os.add_dll_directory(path)
build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise ImportError(
        "Build path not found, please make sure the shared library is built and in the correct path"
    )

from pequegrad_c import *

# grads()
load_cuda_driver_api(True)
