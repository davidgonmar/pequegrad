cmake_minimum_required(VERSION 3.5)
project(pequegrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find the desired Python version
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

# find cuda
find_package(CUDA REQUIRED)

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(csrc)

# Collect all .cu files in the csrc directory
file(GLOB CUDA_SOURCES csrc/*.cu)

# add library from /csrc
pybind11_add_module(pequegrad_cu ${CUDA_SOURCES})

# Set the relocatable device code flag for nvcc
set_property(TARGET pequegrad_cu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(pequegrad_cu PROPERTIES OUTPUT_NAME "pequegrad_cu")