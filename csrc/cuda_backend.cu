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
      .def("sub",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, sub_kernel);
           })
      .def("mul",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, mult_kernel);
           })
      .def("div",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, div_kernel);
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
      .def("where",
           [](const CudaArray &a, const CudaArray &cond, const CudaArray &b) {
             return cond.ternaryop(a, b, where_kernel);
           })
      .def("sum",
           [](const CudaArray &arr, size_t axis) { return arr.sum(axis); })
      .def("sum", [](const CudaArray &arr) { return arr.sum(); })
      .def("sum",
           [](const CudaArray &arr, shape_t axes) { return arr.sum(axes); })
      .def("max",
           [](const CudaArray &arr, size_t axis) { return arr.max(axis); })
      .def("max", [](const CudaArray &arr) { return arr.max(); })
      .def("max",
           [](const CudaArray &arr, shape_t axes) { return arr.max(axes); })
      .def("squeeze", [](const CudaArray &arr, size_t axis) { return arr.squeeze(axis); })
      .def("__getitem__", [](const CudaArray &arr, shape_t index) {
        return arr.getitem(index);
      });
}