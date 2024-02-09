cmake_minimum_required(VERSION 3.12)
project(pequegrad LANGUAGES CXX CUDA)

# find the desired Python version
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# find pybind
find_package(pybind11 REQUIRED HINTS "${Python3_SITELIB}")

# find cuda
find_package(CUDA REQUIRED)

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(csrc)

# Collect all .cu files in the csrc directory
file(GLOB CUDA_SOURCES csrc/*.cu)

# add library from /csrc
pybind11_add_module(pequegrad_cu csrc/cuda_backend.cu)


set_target_properties(pequegrad_cu PROPERTIES OUTPUT_NAME "pequegrad_cu")