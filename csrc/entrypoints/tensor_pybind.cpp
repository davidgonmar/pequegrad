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

  py::enum_<device::DeviceKind>(m, "device")
      .value("cpu", device::DeviceKind::CPU)
      .value("cuda", device::DeviceKind::CUDA);

  // module functions. We need to do the pybind cast overload thing for add
  // since it accepts scalars
  m.def("add", [](const Tensor &a, const Tensor &b) { return pg::add(a, b); });
  m.def("add", [](const Tensor &a, double b) { return pg::add(a, b); });
  m.def("sub", [](const Tensor &a, const Tensor &b) { return pg::sub(a, b); });
  m.def("sub", [](const Tensor &a, double b) { return pg::sub(a, b); });
  m.def("mul", [](const Tensor &a, const Tensor &b) { return pg::mul(a, b); });
  m.def("mul", [](const Tensor &a, double b) { return pg::mul(a, b); });
  m.def("div", [](const Tensor &a, const Tensor &b) { return pg::div(a, b); });
  m.def("div", [](const Tensor &a, double b) { return pg::div(a, b); });
  m.def("neg", &neg);
  m.def("fill", &fill);
  m.def("gt", &gt);
  m.def("lt", &lt);
  m.def("eq", &eq);
  m.def("neq", &neq);
  m.def("pow", [](const Tensor &a, const Tensor &b) { return pg::pow(a, b); });
  m.def("pow", [](const Tensor &a, double b) { return pg::pow(a, b); });
  m.def("log", &pg::log);
  m.def("exp", &pg::exp);
  m.def("max", &pg::max);
  m.def("im2col", &pg::im2col, py::arg("a"), py::arg("kernel_shape"),
        py::arg("stride") = std::vector<int>{1, 1},
        py::arg("padding") = std::vector<int>{0, 0},
        py::arg("dilation") = std::vector<int>{1, 1});
  m.def("col2im", &pg::col2im, py::arg("a"), py::arg("output_shape"),
        py::arg("kernel_shape"), py::arg("stride") = std::vector<int>{1, 1},
        py::arg("padding") = std::vector<int>{0, 0},
        py::arg("dilation") = std::vector<int>{1, 1});
  m.def("reshape", py::overload_cast<const Tensor &, const axes_t &>(&reshape),
        py::arg("a"), py::arg("shape"));

  m.def("unsqueeze", [](const Tensor &a, py::object axes) {
    if (py::isinstance<py::int_>(axes)) {
      return unsqueeze(a, axes.cast<axis_t>());
    } else if (py::isinstance<py::list>(axes) ||
               py::isinstance<py::tuple>(axes)) {
      return unsqueeze(a, axes.cast<axes_t>());
    } else {
      throw std::runtime_error("unsqueeze: axes must be an int, list or tuple");
    }
  });
  m.def("squeeze", [](const Tensor &a, py::object axes) {
    if (py::isinstance<py::int_>(axes)) {
      return squeeze(a, axes.cast<axis_t>());
    } else if (py::isinstance<py::list>(axes) ||
               py::isinstance<py::tuple>(axes)) {
      return squeeze(a, axes.cast<axes_t>());
    } else {
      throw std::runtime_error("squeeze: axes must be an int, list or tuple");
    }
  });
  m.def("broadcast_to", &broadcast_to);
  m.def("broadcast_as", &broadcast_as);

  m.def("matmul", &matmul);
  m.def("where", &where);

#define BIND_REDUCE_OP(python_name, name)                                      \
  m.def(                                                                       \
      python_name,                                                             \
      [](const Tensor &a, py::object axes, bool keepdims) {                    \
        if (axes.is_none()) {                                                  \
          return name(a, keepdims);                                            \
        } else if (py::isinstance<py::int_>(axes)) {                           \
          return name(a, axes.cast<axis_t>(), keepdims);                       \
        } else if (py::isinstance<py::list>(axes) ||                           \
                   py::isinstance<py::tuple>(axes)) {                          \
          return name(a, axes.cast<axes_t>(), keepdims);                       \
        } else {                                                               \
          throw std::runtime_error(#python_name                                \
                                   ": axes must be an int, list, None or "     \
                                   "tuple, and keepdims must be a bool");      \
        }                                                                      \
      },                                                                       \
      py::arg("a"), py::arg("axes") = py::none(),                              \
      py::arg("keepdims") = false);

  BIND_REDUCE_OP("sum", pg::sum);
  BIND_REDUCE_OP("max_reduce", pg::max_reduce);
  BIND_REDUCE_OP("mean", pg::mean);

  m.def("permute", &permute);

  m.def("grads", [](std::vector<Tensor> required_tensors, const Tensor &output,
                    std::optional<Tensor> tangent) {
    return grads(required_tensors, output, tangent);
  });
  m.def("grads",
        [](std::vector<Tensor> required_tensors, const Tensor &output) {
          return grads(required_tensors, output);
        });
  // module classes
  py::class_<Tensor>(m, "Tensor")
      .def("to_", &Tensor::to_)
      .def("assign", &Tensor::assign)
      .def("detach", &Tensor::detach)
      .def("detach_", &Tensor::detach_)
      .def("children", &Tensor::children)
      .def("ad_context",
           [](const Tensor &t) { return t.ad_node().primitive()->str(); })
      .def("to", &Tensor::to)
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
               throw std::runtime_error("Unsupported data type: " +
                                        dtype_to_string(arr.dtype()));
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
               throw std::runtime_error("Unsupported data type: " +
                                        dtype_to_string(arr.dtype()));
             }
           })
      .def_property_readonly("shape", [](const Tensor &t) { return t.shape(); })
      .def_property_readonly("dtype", [](const Tensor &t) { return t.dtype(); })
      .def_property_readonly("device",
                             [](const Tensor &t) { return t.device(); })
      .def(py::init([](py::array_t<float> np_array, bool requires_grad,
                       device::DeviceKind device) {
             return Tensor::from_numpy(np_array, requires_grad, device);
           }),
           py::arg("np_array"), py::arg("requires_grad") = false,
           py::arg("device") = device::DeviceKind::CPU)
      .def("__add__",
           [](const Tensor &a, const Tensor &b) { return pg::add(a, b); })
      .def("__add__", [](const Tensor &a, double b) { return pg::add(a, b); })
      .def("__add__", [](const Tensor &a, float b) { return pg::add(a, b); })
      .def("__radd__", [](const Tensor &a, double b) { return pg::add(a, b); })
      .def("__radd__", [](const Tensor &a, float b) { return pg::add(a, b); })
      .def("__radd__",
           [](const Tensor &a, const Tensor &b) { return pg::add(a, b); })
      .def("__sub__",
           [](const Tensor &a, const Tensor &b) { return pg::sub(a, b); })
      .def("__sub__", [](const Tensor &a, double b) { return pg::sub(a, b); })
      .def("__sub__", [](const Tensor &a, float b) { return pg::sub(a, b); })
      .def("__sub__", [](const Tensor &a, int b) { return pg::sub(a, b); })
      .def("__mul__",
           [](const Tensor &a, const Tensor &b) { return pg::mul(a, b); })
      .def("__mul__", [](const Tensor &a, double b) { return pg::mul(a, b); })
      .def("__mul__", [](const Tensor &a, float b) { return pg::mul(a, b); })
      .def("__mul__", [](const Tensor &a, int b) { return pg::mul(a, b); })
      .def("__rmul__", [](const Tensor &a, double b) { return pg::mul(a, b); })
      .def("__rmul__", [](const Tensor &a, float b) { return pg::mul(a, b); })
      .def("__rmul__", [](const Tensor &a, int b) { return pg::mul(a, b); })
      .def("__truediv__",
           [](const Tensor &a, const Tensor &b) { return pg::div(a, b); })
      .def("__truediv__",
           [](const Tensor &a, double b) { return pg::div(a, b); })
      .def("__neg__", [](const Tensor &a) { return pg::neg(a); })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return pg::matmul(a, b); })
      .def("__pow__",
           [](const Tensor &a, const Tensor &b) { return pg::pow(a, b); })
      .def("__pow__", [](const Tensor &a, double b) { return pg::pow(a, b); })
      .def("__pow__", [](const Tensor &a, float b) { return pg::pow(a, b); })
      .def("__pow__", [](const Tensor &a, int b) { return pg::pow(a, b); })
      .def("__eq__",
           [](const Tensor &a, const Tensor &b) { return pg::eq(a, b); })
      .def("__ne__",
           [](const Tensor &a, const Tensor &b) { return pg::neq(a, b); })
      .def("__lt__",
           [](const Tensor &a, const Tensor &b) { return pg::lt(a, b); })
      .def("__gt__",
           [](const Tensor &a, const Tensor &b) { return pg::gt(a, b); })
      .def("__repr__", [](const Tensor &t) {
        std::stringstream ss;
        ss << "Tensor(shape=" << vec_to_string(t.shape())
           << ", dtype=" << dtype_to_string(t.dtype())
           << ", device=" << t.device() << ", evaled=" << t.is_evaled() << ")";
        return ss.str();
      });
};