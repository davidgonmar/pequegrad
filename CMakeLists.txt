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

file(GLOB CUDA_SOURCES 
    csrc/dtype.cpp
    csrc/shape.cpp
    csrc/pybind_utils.cu
    csrc/utils.cu
    csrc/cuda_tensor/*.cu 
    csrc/cuda_tensor/*.cpp 
    csrc/cuda_tensor/**/*.cu  
    csrc/entrypoints/pybind_entrypoint.cu
)

# add library from /csrc
pybind11_add_module(pequegrad_cu ${CUDA_SOURCES})

# Set the relocatable device code flag for nvcc
set_property(TARGET pequegrad_cu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(pequegrad_cu PROPERTIES OUTPUT_NAME "pequegrad_cu")




# ========================== CPU version ==========================

# libopenblas.dll MUST BE SOMEWHERE SEARCHABLE BY WINDOWS (for example, in the same directory as the executable, or under System32)
# Specify path to the Cmake file of the BLAS library
set(BLAS_INCLUDE_DIR "C:/OpenBLAS/include")

# check if the BLAS include directory exists
if(NOT EXISTS ${BLAS_INCLUDE_DIR})
    message(FATAL_ERROR "BLAS include directory does not exist: ${BLAS_INCLUDE_DIR}")
else()
    # Print the include directories
    message(STATUS "BLAS include directories: ${BLAS_INCLUDE_DIR}")
    include_directories(${BLAS_INCLUDE_DIR})
endif()

set(BLAS_LIB "C:/OpenBLAS/lib/libopenblas.lib")

# check if the BLAS library exists
if(NOT EXISTS ${BLAS_LIB})
    message(FATAL_ERROR "BLAS library does not exist: ${BLAS_LIB}")
else()
    # Print the library
    message(STATUS "BLAS library: ${BLAS_LIB}")
endif()

file(GLOB CPU_SOURCES 
    csrc/dtype.cpp
    csrc/shape.cpp
    csrc/pybind_utils.cu
    csrc/utils.cu
    csrc/cpu_tensor/*.cpp 
    csrc/cpu_tensor/**/*.cu 
    csrc/entrypoints/pybind_entrypoint.cpp
)
# Add an executable target
pybind11_add_module(pequegrad_cpu ${CPU_SOURCES})
# link the BLAS library

target_link_libraries(pequegrad_cpu PRIVATE ${BLAS_LIB})


# Set C++ standard
set_property(TARGET pequegrad_cpu PROPERTY CXX_STANDARD 17)

set_target_properties(pequegrad_cpu PROPERTIES OUTPUT_NAME "pequegrad_cpu") 