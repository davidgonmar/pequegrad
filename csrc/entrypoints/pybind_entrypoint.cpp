#include "cpu_tensor/cpu_tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

namespace py = pybind11;
using ItemVariant = std::variant<float, int, double>;
using NpArrayVariant =
    std::variant<py::array_t<float>, py::array_t<int>, py::array_t<double>>;

#define BIND_BINARY_OP(op)                                                     \
  def(#op, [](const CpuTensor &a, const CpuTensor &b) { return a.op(b); })

PYBIND11_MODULE(pequegrad_cpu, m) {
  namespace py = pybind11;

  m.attr("__device_name__") = "cpu";
  py::class_<CpuTensor>(m, "CpuTensor")
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return CpuTensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<int> np_array) {
             return CpuTensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<double> np_array) {
             return CpuTensor::from_numpy(np_array);
           })
      .def("to_numpy",
           [](const CpuTensor &arr) -> NpArrayVariant {
             switch (arr.dtype) {
             case DType::Float32:
               return arr.to_numpy<float>();
             case DType::Int32:
               return arr.to_numpy<int>();
             case DType::Float64:
               return arr.to_numpy<double>();
             default:
               throw std::runtime_error("Unsupported data type");
             }
           })
      .def("add",
           [](const CpuTensor &a, const CpuTensor &b) { return a.add(b); })
      .def_property_readonly("dtype",
                             [](const CpuTensor &arr) {
                               switch (arr.dtype) {
                               case DType::Float32:
                                 return "float32";
                               case DType::Int32:
                                 return "int32";
                               case DType::Float64:
                                 return "float64";
                               default:
                                 throw std::runtime_error(
                                     "Unsupported data type");
                               }
                             })
      .def_property_readonly("shape",
                             [](const CpuTensor &arr) {
                               py::tuple t(arr.shape.size());
                               for (size_t i = 0; i < arr.shape.size(); i++) {
                                 t[i] = arr.shape[i];
                               }
                               return t;
                             })
      .def_property_readonly(
          "ndim", [](const CpuTensor &arr) { return arr.shape.size(); })
      .def_property_readonly("strides",
                             [](const CpuTensor &arr) {
                               py::tuple t(arr.strides.size());
                               for (size_t i = 0; i < arr.strides.size(); i++) {
                                 t[i] = arr.strides[i];
                               }
                               return t;
                             })
      .def_property_readonly("nbytes",
                             [](const CpuTensor &arr) { return arr.nbytes; })
      .def("log", [](const CpuTensor &arr) { return arr.log(); })
      .def("exp", [](const CpuTensor &arr) { return arr.exp(); })
      .BIND_BINARY_OP(sub)
      .BIND_BINARY_OP(mul)
      .BIND_BINARY_OP(div)
      .BIND_BINARY_OP(gt)
      .BIND_BINARY_OP(lt)
      .BIND_BINARY_OP(eq)
      .BIND_BINARY_OP(ne)
      .BIND_BINARY_OP(ge)
      .BIND_BINARY_OP(le)
      .BIND_BINARY_OP(pow)
      .BIND_BINARY_OP(el_wise_max)
      .def("broadcast_to",
           [](const CpuTensor &arr, const std::vector<size_t> &new_shape) {
             return arr.broadcast_to(new_shape);
           })
      .def(
          "squeeze",
          [](const CpuTensor &arr, axis_t axis) { return arr.squeeze(axis); },
          py::arg("axis"))
      .def(
          "squeeze",
          [](const CpuTensor &arr, axes_t axes) { return arr.squeeze(axes); },
          py::arg("axes"))
      .def("squeeze", [](const CpuTensor &arr) { return arr.squeeze(); })
      .def("unsqueeze", [](const CpuTensor &arr,
                           axis_t axis) { return arr.unsqueeze(axis); })
      .def("unsqueeze", [](const CpuTensor &arr,
                           axes_t axes) { return arr.unsqueeze(axes); })
      .def("reshape",
           [](const CpuTensor &arr, py::args new_shape) {
             std::vector<int> shape_vec = new_shape.cast<std::vector<int>>();
             return arr.reshape(shape_vec);
           })
      .def("permute",
           [](const CpuTensor &arr, py::args axes) {
             shape_t axes_vec = axes.cast<shape_t>();
             return arr.permute(axes_vec);
           })
      .def("transpose",
           [](const CpuTensor &arr) {
             if (arr.ndim() < 2) {
               return arr;
             }
             shape_t axes(arr.ndim());
             std::iota(axes.begin(), axes.end(), 0);
             std::swap(axes[0], axes[1]);
             return arr.permute(axes);
           })
      .def("contiguous",
           [](const CpuTensor &arr) { return arr.as_contiguous(); })
      .def("is_contiguous",
           [](const CpuTensor &arr) { return arr.is_contiguous(); });
}
