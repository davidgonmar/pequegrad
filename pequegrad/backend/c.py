import os
import sys

# todo -- it cannot find the CUDA dlls for nvrtc if I don't add the path here
paths = [
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin",
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
