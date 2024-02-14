#include "binary_ops_kernels.cuh"
#include "cuda_array.cuh"
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

namespace py = pybind11;

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;

  m.attr("__device_name__") = "cuda";
  py::class_<CudaArray>(m, "Array")
      .def_readonly("size", &CudaArray::size)
      .def_readonly("shape", &CudaArray::shape)
      .def_readonly("strides", &CudaArray::strides)
      .def("clone", &CudaArray::clone)
      .def("broadcast_to", &CudaArray::broadcast_to)
      .def("to_numpy", &CudaArray::to_numpy)
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return CudaArray::from_numpy(np_array);
           })
      .def("__repr__", [](const CudaArray &arr) { return arr.to_string(); })
      .def("add",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, add_kernel);
           })
      .def("add",
           [](const CudaArray &arr, const py::array_t<float> np_array) {
             return arr.binop(np_array, add_kernel);
           })
      .def("sub",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, sub_kernel);
           })
      .def("sub",
           [](const CudaArray &arr, const py::array_t<float> np_array) {
             return arr.binop(np_array, sub_kernel);
           })
      .def("mul",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, mult_kernel);
           })
      .def("mul",
           [](const CudaArray &arr, const py::array_t<float> np_array) {
             return arr.binop(np_array, mult_kernel);
           })
      .def("div",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, div_kernel);
           })
      .def("div",
           [](const CudaArray &arr, const py::array_t<float> np_array) {
             return arr.binop(np_array, div_kernel);
           })

      .def("el_wise_max",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, element_wise_max_kernel);
           })

      .def("pow",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, pow_kernel);
           })
      .def("eq",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, equal_kernel);
           })
      .def("ne",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, not_equal_kernel);
           })
      .def("lt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, less_kernel);
           })
      .def("le",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, less_equal_kernel);
           })
      .def("gt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, greater_kernel);
           })
      .def("ge",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, greater_equal_kernel);
           })
      .def("contiguous", &CudaArray::as_contiguous)
      .def("exp", [](const CudaArray &arr) { return arr.elwiseop(exp_kernel); })
      .def("log", [](const CudaArray &arr) { return arr.elwiseop(log_kernel); })
      .def("permute",
           [](const CudaArray &arr, shape_t axes) { return arr.permute(axes); })
      .def("is_contiguous", &CudaArray::is_contiguous)
      .def("matmul", [](const CudaArray &arr,
                        const CudaArray &other) { return arr.mat_mul(other); })
      .def("outer_product",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.outer_product(other);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond, const CudaArray &b) {
             return cond.ternaryop(a, b, where_kernel);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<float> np_array) {
             return cond.ternaryop(a, np_array, where_kernel);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<float> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, where_kernel);
           })
      .def("sum", [](const CudaArray &arr, axis_t axis,
                     bool keepdims) { return arr.sum(axis, keepdims); })
      .def("sum", [](const CudaArray &arr,
                     bool keepdims) { return arr.sum(keepdims); })
      .def("sum", [](const CudaArray &arr, axes_t axes,
                     bool keepdims) { return arr.sum(axes, keepdims); })
      .def("max", [](const CudaArray &arr,
                     bool keepdims) { return arr.max(keepdims); })
      .def("max", [](const CudaArray &arr, axis_t axis,
                     bool keepdims) { return arr.max(axis, keepdims); })
      .def("max", [](const CudaArray &arr, axes_t axes,
                     bool keepdims) { return arr.max(axes, keepdims); })
      .def("squeeze",
           [](const CudaArray &arr, axis_t axis) { return arr.squeeze(axis); })
      .def("unsqueeze", [](const CudaArray &arr,
                           axis_t axis) { return arr.unsqueeze(axis); })
      .def("unsqueeze", [](const CudaArray &arr,
                           axes_t axes) { return arr.unsqueeze(axes); })
      .def("reshape",
           [](const CudaArray &arr, std::vector<int> new_shape) {
             return arr.reshape(new_shape);
           })
      .def("__getitem__", [](const CudaArray &arr, shape_t index) {
        return arr.getitem(index);
      });
}