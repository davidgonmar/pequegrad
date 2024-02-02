cmake_minimum_required(VERSION 3.12)
project(pequegrad LANGUAGES CXX CUDA)

# find the desired Python version
find_package(PythonLibs EXACT 3.10 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

# find cuda
find_package(CUDA REQUIRED)

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})

# add library from /csrc
pybind11_add_module(pequegrad_cu csrc/cuda_backend.cu)


set_target_properties(pequegrad_cu PROPERTIES OUTPUT_NAME "pequegrad_cu")