#include "binary_ops_kernels.cuh"
#include "cuda_array.cuh"
#include "cuda_array_impl.cuh"
#include "matmul_kernels.cuh"
#include "ternary_ops_kernels.cuh"
#include "unary_ops_kernels.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

#define BIND_CUDARRAY_PY(T, NAME)                                              \
  py::class_<CudaArray<T>>(m, NAME)                                            \
      .def_readonly("shape", &CudaArray<T>::shape)                             \
      .def_readonly("strides", &CudaArray<T>::strides)                         \
      .def("clone", &CudaArray<T>::clone)                                      \
      .def("broadcast_to", &CudaArray<T>::broadcast_to)                        \
      .def("to_numpy", &CudaArray<T>::to_numpy)                                \
      .def("from_numpy",                                                       \
           [](py::array_t<T> np_array) {                                       \
             return CudaArray<T>::from_numpy(np_array);                        \
           })                                                                  \
      .def("__repr__",                                                         \
           [](const CudaArray<T> &arr) { return arr.to_string(); })            \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<int> &other) {           \
            return arr.binop(other, add_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<int> np_array) {       \
            return arr.binop(np_array, add_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<float> &other) {         \
            return arr.binop(other, add_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<float> np_array) {     \
            return arr.binop(np_array, add_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<double> &other) {        \
            return arr.binop(other, add_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<double> np_array) {    \
            return arr.binop(np_array, add_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, int scalar) {                            \
            return arr.binop(scalar, add_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, float scalar) {                          \
            return arr.binop(scalar, add_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "add",                                                               \
          [](const CudaArray<T> &arr, double scalar) {                         \
            return arr.binop(scalar, add_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<int> &other) {           \
            return arr.binop(other, sub_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<int> np_array) {       \
            return arr.binop(np_array, sub_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<float> &other) {         \
            return arr.binop(other, sub_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<float> np_array) {     \
            return arr.binop(np_array, sub_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<double> &other) {        \
            return arr.binop(other, sub_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<double> np_array) {    \
            return arr.binop(np_array, sub_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, int scalar) {                            \
            return arr.binop(scalar, sub_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, float scalar) {                          \
            return arr.binop(scalar, sub_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "sub",                                                               \
          [](const CudaArray<T> &arr, double scalar) {                         \
            return arr.binop(scalar, sub_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def("mul",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<int> &other) {          \
             return arr.binop(other, mult_kernel);                             \
           })                                                                  \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<float> np_array) {     \
            return arr.binop(np_array, mult_kernel);                           \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def("mul",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<float> &other) {        \
             return arr.binop(other, mult_kernel);                             \
           })                                                                  \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<int> np_array) {       \
            return arr.binop(np_array, mult_kernel);                           \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def("mul",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<double> &other) {       \
             return arr.binop(other, mult_kernel);                             \
           })                                                                  \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<double> np_array) {    \
            return arr.binop(np_array, mult_kernel);                           \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const double scalar) {                   \
            return arr.binop(scalar, mult_kernel);                             \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const float scalar) {                    \
            return arr.binop(scalar, mult_kernel);                             \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "mul",                                                               \
          [](const CudaArray<T> &arr, const int scalar) {                      \
            return arr.binop(scalar, mult_kernel);                             \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<int> &other) {           \
            return arr.binop(other, div_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<int> np_array) {       \
            return arr.binop(np_array, div_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<float> &other) {         \
            return arr.binop(other, div_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<float> np_array) {     \
            return arr.binop(np_array, div_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const CudaArray<double> &other) {        \
            return arr.binop(other, div_kernel);                               \
          },                                                                   \
          py::arg("other").noconvert())                                        \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const py::array_t<double> np_array) {    \
            return arr.binop(np_array, div_kernel);                            \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const int scalar) {                      \
            return arr.binop(scalar, div_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const float scalar) {                    \
            return arr.binop(scalar, div_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          "div",                                                               \
          [](const CudaArray<T> &arr, const double scalar) {                   \
            return arr.binop(scalar, div_kernel);                              \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def("el_wise_max",                                                      \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, element_wise_max_kernel);                 \
           })                                                                  \
      .def("pow",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<int> &other) {          \
             return arr.binop(other, pow_kernel);                              \
           })                                                                  \
      .def("pow",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<float> &other) {        \
             return arr.binop(other, pow_kernel);                              \
           })                                                                  \
      .def("pow",                                                              \
           [](const CudaArray<T> &arr, const CudaArray<double> &other) {       \
             return arr.binop(other, pow_kernel);                              \
           })                                                                  \
      .def("eq",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, equal_kernel);                            \
           })                                                                  \
      .def("ne",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, not_equal_kernel);                        \
           })                                                                  \
      .def("lt",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, less_kernel);                             \
           })                                                                  \
      .def("le",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, less_equal_kernel);                       \
           })                                                                  \
      .def("gt",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, greater_kernel);                          \
           })                                                                  \
      .def("ge",                                                               \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.binop(other, greater_equal_kernel);                    \
           })                                                                  \
      .def("contiguous", &CudaArray<T>::as_contiguous)                         \
      .def("exp",                                                              \
           [](const CudaArray<T> &arr) { return arr.elwiseop(exp_kernel); })   \
      .def("log",                                                              \
           [](const CudaArray<T> &arr) { return arr.elwiseop(log_kernel); })   \
      .def("permute", [](const CudaArray<T> &arr,                              \
                         shape_t axes) { return arr.permute(axes); })          \
      .def("is_contiguous", &CudaArray<T>::is_contiguous)                      \
      .def("matmul",                                                           \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.mat_mul(other);                                        \
           })                                                                  \
      .def("outer_product",                                                    \
           [](const CudaArray<T> &arr, const CudaArray<T> &other) {            \
             return arr.outer_product(other);                                  \
           })                                                                  \
      .def("where",                                                            \
           [](const CudaArray<T> &a, const CudaArray<T> &cond,                 \
              const CudaArray<T> &b) {                                         \
             return cond.ternaryop(a, b, where_kernel);                        \
           })                                                                  \
      .def("where",                                                            \
           [](const CudaArray<T> &a, const CudaArray<T> &cond,                 \
              const py::array_t<T> np_array) {                                 \
             return cond.ternaryop(a, np_array, where_kernel);                 \
           })                                                                  \
      .def("where",                                                            \
           [](const CudaArray<T> &a, const py::array_t<T> np_array,            \
              const CudaArray<T> &b) {                                         \
             return a.ternaryop(np_array, b, where_kernel);                    \
           })                                                                  \
      .def("sum", [](const CudaArray<T> &arr, axis_t axis,                     \
                     bool keepdims) { return arr.sum(axis, keepdims); })       \
      .def("sum", [](const CudaArray<T> &arr,                                  \
                     bool keepdims) { return arr.sum(keepdims); })             \
      .def("sum", [](const CudaArray<T> &arr, axes_t axes,                     \
                     bool keepdims) { return arr.sum(axes, keepdims); })       \
      .def("max", [](const CudaArray<T> &arr,                                  \
                     bool keepdims) { return arr.max(keepdims); })             \
      .def("max", [](const CudaArray<T> &arr, axis_t axis,                     \
                     bool keepdims) { return arr.max(axis, keepdims); })       \
      .def("max", [](const CudaArray<T> &arr, axes_t axes,                     \
                     bool keepdims) { return arr.max(axes, keepdims); })       \
      .def("squeeze", [](const CudaArray<T> &arr,                              \
                         axis_t axis) { return arr.squeeze(axis); })           \
      .def("squeeze", [](const CudaArray<T> &arr,                              \
                         axes_t axes) { return arr.squeeze(axes); })           \
      .def("squeeze", [](const CudaArray<T> &arr) { return arr.squeeze(); })   \
      .def("unsqueeze", [](const CudaArray<T> &arr,                            \
                           axis_t axis) { return arr.unsqueeze(axis); })       \
      .def("unsqueeze", [](const CudaArray<T> &arr,                            \
                           axes_t axes) { return arr.unsqueeze(axes); })       \
      .def("reshape",                                                          \
           [](const CudaArray<T> &arr, std::vector<int> new_shape) {           \
             return arr.reshape(new_shape);                                    \
           })                                                                  \
      .def("im2col",                                                           \
           [](const CudaArray<T> &arr, shape_t kernel_shape, size_t stride) {  \
             return arr.im2col(kernel_shape, stride);                          \
           })                                                                  \
      .def("col2im",                                                           \
           [](const CudaArray<T> &arr, shape_t kernel_shape,                   \
              shape_t output_shape, size_t stride) {                           \
             return arr.col2im(kernel_shape, output_shape, stride);            \
           })                                                                  \
      .def_static("fill", &CudaArray<T>::fill)                                 \
      .def("__getitem__", [](const CudaArray<T> &arr, shape_t index) {         \
        return arr.getitem(index);                                             \
      });

namespace py = pybind11;
// first declare that there exists CudaArray<T> for each type (not bindings)

template class CudaArray<int>;
template class CudaArray<float>;
template class CudaArray<double>;

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;

  m.attr("__device_name__") = "cuda";

  BIND_CUDARRAY_PY(float, "CudaArrayFloat32");  // float32
  BIND_CUDARRAY_PY(double, "CudaArrayFloat64"); // float64
  BIND_CUDARRAY_PY(int, "CudaArrayInt32");      // int32
}