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
      .def("broadcast_to", &CudaArray::broadcastTo)
      .def("to_numpy", &CudaArray::toNumpy)
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return CudaArray::fromNumpy(np_array);
           })
      .def("__repr__", [](const CudaArray &arr) { return arr.toString(); })
      .def("add",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, AddKernel);
           })
      .def("sub",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, SubKernel);
           })
      .def("mul",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, MultKernel);
           })
      .def("div",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, DivKernel);
           })
      .def("el_wise_max",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, ElementWiseMaxKernel);
           })
      .def("pow",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, PowKernel);
           })
      .def("eq",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, EqualKernel);
           })
      .def("ne",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, NotEqualKernel);
           })
      .def("lt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, LessKernel);
           })
      .def("le",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, LessEqualKernel);
           })
      .def("gt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, GreaterKernel);
           })
      .def("ge",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, GreaterEqualKernel);
           })
      .def("contiguous",
           [](const CudaArray &arr) { return arr.asContiguous(); })
      .def("exp", [](const CudaArray &arr) { return arr.elwiseop(ExpKernel); })
      .def("log", [](const CudaArray &arr) { return arr.elwiseop(LogKernel); })
      .def("permute", [](const CudaArray &arr,
                         ShapeLike axes) { return arr.permute(axes); })
      .def("is_contiguous", &CudaArray::isContiguous)
      .def("matmul", [](const CudaArray &arr,
                        const CudaArray &other) { return arr.matMul(other); })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond, const CudaArray &b) {
             return cond.ternaryop(a, b, WhereKernel);
           })
      .def("__getitem__", [](const CudaArray &arr, ShapeLike index) {
        return arr.getitem(index);
      });
}