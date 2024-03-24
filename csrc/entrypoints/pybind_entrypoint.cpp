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
                             [](const CpuTensor &arr) { return arr.shape; })
      .def_property_readonly("strides",
                             [](const CpuTensor &arr) { return arr.strides; })
      .def_property_readonly("nbytes",
                             [](const CpuTensor &arr) { return arr.nbytes; })
      .def("log", [](const CpuTensor &arr) { return arr.log(); })
      .def("exp", [](const CpuTensor &arr) { return arr.exp(); });
}
