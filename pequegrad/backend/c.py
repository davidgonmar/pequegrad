import os
import sys

build_path = os.path.join(os.path.dirname(__file__), "..", "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise ImportError("Build path not found, please run `make` in the root directory")

from pequegrad_c import *  # noqa
