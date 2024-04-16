#include "dtype.hpp"
#include "ops.hpp"
#include "pybind_utils.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <stdio.h>
#include <string>

namespace py = pybind11;
using namespace pg;
using ItemVariant = std::variant<float, int, double>;
using NpArrayVariant =
    std::variant<py::array_t<float>, py::array_t<int>, py::array_t<double>>;

#define BIND_BINARY_OP(op)                                                     \
  def(#op, [](const Tensor &a, const Tensor &b) { return a.op(b); })

PYBIND11_MODULE(pequegrad_c, m) {
  namespace py = pybind11;

  py::enum_<DType>(m, "dt")
      .value("float32", DType::Float32)
      .value("int32", DType::Int32)
      .value("float64", DType::Float64);
  
  // module functions
  m.def("add", &add);
  m.def("mul", &mul);
  m.def("sub", &sub);
  m.def("div", &pg::div);
  m.def("neg", &neg);
  m.def("fill", &fill);
  m.def("gt", &gt);
  m.def("lt", &lt);
  m.def("eq", &eq);
  m.def("neq", &neq);
  m.def("pow", &pg::pow);
  m.def("log", &pg::log);


  // module classes
  py::class_<Tensor>(m, "Tensor")
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return Tensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<int> np_array) {
             return Tensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<double> np_array) {
             return Tensor::from_numpy(np_array);
           })
      .def("eval", &Tensor::eval)
      .def("to_numpy",
           [](Tensor &arr) -> NpArrayVariant {
             if (!arr.is_evaled())
               arr.eval();

             switch (arr.dtype()) {
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
      .def("numpy",
           [](Tensor &arr) -> NpArrayVariant {
             if (!arr.is_evaled())
               arr.eval();

             switch (arr.dtype()) {
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
      .def("backward", &Tensor::backward)
      .def_property_readonly("grad", [](const Tensor &t) { return t.grad(); })
      .def_property_readonly("shape", [](const Tensor &t) { return t.shape(); });
};