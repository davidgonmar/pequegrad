cmake_minimum_required(VERSION 3.17)
project(pequegrad LANGUAGES CXX)

include_directories(external/cutlass/include)
include_directories(external/cudnn-frontend/include)

# Find the desired Python version
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Find pybind
execute_process(COMMAND python -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(pybind11 PATHS ${__pybind_path})
include_directories(csrc)


# Find OpenBLAS
# libopenblas.dll MUST BE SOMEWHERE SEARCHABLE BY WINDOWS (for example, in the same directory as the executable, or under System32)
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

# Check if the BLAS library exists
if(NOT EXISTS ${BLAS_LIB})
    message(FATAL_ERROR "BLAS library does not exist: ${BLAS_LIB}")
else()
    message(STATUS "BLAS library: ${BLAS_LIB}")
endif()

include(CheckCXXCompilerFlag)


### Instead of doing target_compile_options, we set the cxx_flags variable and pass it to the target_compile_options
### But only for the CPU version, as the CUDA compiler does not like those flags
enable_language(CUDA)

find_package(CUDA ${CUDA_VERSION} REQUIRED)
find_package(CUDAToolkit ${CUDA_VERSION} REQUIRED)

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
# cutlass

file(GLOB PG_SOURCES
        csrc/dtype.cpp
        csrc/shape.cpp
        csrc/common/*.cpp
        csrc/pybind_utils.cpp
        csrc/utils.cu
        csrc/cuda_tensor/*.cu 
        csrc/cuda_tensor/*.cpp 
        csrc/cuda_tensor/**/*.cu  
        csrc/cpu/*.cpp
        csrc/cuda/*.cu
        csrc/cuda/*.cpp
        csrc/entrypoints/tensor_pybind.cpp
        csrc/entrypoints/npybind_utils.cpp
        csrc/*.cpp
        csrc/compiler/*.cpp
    )

pybind11_add_module(pequegrad_c ${PG_SOURCES})

set_property(TARGET pequegrad_c PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET pequegrad_c PROPERTY CXX_STANDARD 17)
set_property(TARGET pequegrad_c PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(pequegrad_c PROPERTIES OUTPUT_NAME "pequegrad_c")

target_link_libraries(pequegrad_c PRIVATE ${BLAS_LIB})
# link to nvrtc
# but first assert that nvrtc is available
if(NOT CUDA_nvrtc_LIBRARY)
    message(FATAL_ERROR "nvrtc library not found")
endif()

target_link_libraries(pequegrad_c PRIVATE CUDA::nvrtc)
target_link_libraries(pequegrad_c PRIVATE CUDA::cublas)

# use OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(pequegrad_c PRIVATE OpenMP::OpenMP_CXX)
endif()

# add max optimization
if (MSVC)
    set(cxx_flags "/Ox")
    set(cu_flags "-O3")
else()
    set(cxx_flags "-O3")
    set(cu_flags "-O3")
endif()

if (MSVC)
    # add to cxx flags
    set(cxx_flags "${cxx_flags} /arch:AVX2")
else()
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)

        set(cxx_flags "${cxx_flags} -mavx2")
    endif()
endif()

if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8380")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /STACK:8388")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /STACK:8388")
endif()

# CUDNN
# Manually specify paths for cuDNN
set(CUDNN_INCLUDE_DIR "C:\\Program Files\\NVIDIA\\CUDNN\\v9.2\\include\\12.5")
set(CUDNN_LIB "C:\\Program Files\\NVIDIA\\CUDNN\\v9.2\\lib\\12.5\\x64\\cudnn.lib")



# Add include directories for cuDNN
include_directories(${CUDNN_INCLUDE_DIR})


# Link cuDNN library with your target
target_link_libraries(pequegrad_c PRIVATE ${CUDNN_LIB})

target_compile_options(pequegrad_c PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>)
target_compile_options(pequegrad_c PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${cu_flags}>)


