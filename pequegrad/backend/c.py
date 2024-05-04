import os
import sys


build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise ImportError(
        "Build path not found, please make sure the shared library is built and in the correct path"
    )

from pequegrad_c import *
