import os
import sys

build_path = os.path.join(os.path.dirname(__file__), "..", "build")
if os.path.exists(build_path):
    sys.path.append(build_path)

# todo - make it output somewhere else and make it general
windows_vs_build_path = os.path.join(
    os.path.dirname(__file__), "..", "out", "build", "x64-Debug"
)

if os.path.exists(windows_vs_build_path):
    sys.path.append(windows_vs_build_path)


# print what python can find

print(sys.path)


import pequegrad_cu  # noqa: F401
