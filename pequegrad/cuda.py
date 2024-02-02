import os
import sys

build_path = os.path.join(os.path.dirname(__file__), "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)
