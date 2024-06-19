#include "compiler/compile.hpp"
#include "dtype.hpp"
#include "graph.hpp"
#include "npybind_utils.hpp"
#include "ops.hpp"
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
#define BIND_BINOP_WITH_SCALAR_OVERLOADS(op)                                   \
  m.def(#op, py::overload_cast<const Tensor &, const Tensor &>(&pg::op));      \
  m.def(#op, py::overload_cast<const Tensor &, double>(&pg::op));              \
  m.def(#op, py::overload_cast<double, const Tensor &>(&pg::op))

  m.def("clone_graph", clone_graph);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(add);
  m.def("add_inplace", [](Tensor &a, const Tensor &b) { add_inplace(a, b); });
  BIND_BINOP_WITH_SCALAR_OVERLOADS(sub);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(mul);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(div);
  m.def("neg", &neg);
  m.def("fill", &fill);

  BIND_BINOP_WITH_SCALAR_OVERLOADS(lt);

  BIND_BINOP_WITH_SCALAR_OVERLOADS(gt);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(eq);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(neq);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(max);
  BIND_BINOP_WITH_SCALAR_OVERLOADS(pow);

  m.def("log", &pg::log);
  m.def("exp", &pg::exp);
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
  m.def("assign_at", [](const Tensor &dst, const Tensor &src,
                        const py::tuple slices) {
    std::vector<hl_select_t> parsed =
        pybind_utils::parse_pybind_slices(slices, dst.shape(), dst.device());
    return assign_at(dst, src, parsed);
  });

  m.def("assign_at", [](const Tensor &dst, const Tensor &src,
                        const pybind_utils::pybind_slice_item_t sl) {
    auto _tuple = py::make_tuple(sl);
    std::vector<hl_select_t> parsed =
        pybind_utils::parse_pybind_slices(_tuple, dst.shape(), dst.device());
    return assign_at(dst, src, parsed);
  });
  m.def("astype", &astype);
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
  m.def("compile", &compile);

  m.def(
      "grads",
      [](std::vector<Tensor> required_tensors, const Tensor &output,
         std::optional<Tensor> tangent) {
        return grads(required_tensors, output, tangent);
      },
      py::arg("required_tensors"), py::arg("output"),
      py::arg("tangent") = std::nullopt);

  m.def("load_cuda_driver_api", [](bool x) {
    cuDevicePrimaryCtxRetain(0, 0); // This is a dummy call to load the driver
    cuInit(0);
    return true;
  });

  m.def("binomial", &binomial, py::arg("p"), py::arg("shape"), py::arg("dtype"),
        py::arg("device") = device::DeviceKind::CPU);

#define BIND_BINOP_WITH_OVERLOAD_CLASS(pyname, op)                             \
  def(#pyname, py::overload_cast<const Tensor &, const Tensor &>(&pg::op))     \
      .def(#pyname, py::overload_cast<const Tensor &, double>(&pg::op))        \
      .def(#pyname, py::overload_cast<double, const Tensor &>(&pg::op))

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
      .def(
          "eval", [](Tensor &t, bool detach) { return t.eval(detach); },
          py::arg("detach") = true)
      .def("numel", &Tensor::numel)
      .def("astype", &Tensor::astype)
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
      .def("is_evaled", &Tensor::is_evaled)
      .def("__hash__", [](const Tensor &t) { return t.id; })
      .def_property_readonly("id", [](const Tensor &t) { return t.id; })
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
      .def(py::init([](py::array_t<float> np_array, device::DeviceKind device) {
             return Tensor::from_numpy(np_array, device);
           }),
           py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)
      .def(py::init([](py::array_t<int> np_array, device::DeviceKind device) {
             return Tensor::from_numpy(np_array, device);
           }),
           py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)
      .def(
          py::init([](py::array_t<double> np_array, device::DeviceKind device) {
            return Tensor::from_numpy(np_array, device);
          }),
          py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)

      .BIND_BINOP_WITH_OVERLOAD_CLASS(add, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__add__, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__radd__, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(sub, sub)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__sub__, sub)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__rsub__, sub)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(mul, mul)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__mul__, mul)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__rmul__, mul)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(div, div)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__truediv__, div)
      .def("__rtruediv__",
           [](const Tensor &a, const Tensor &b) { return pg::div(b, a); })
      .def("__rtruediv__",
           [](const Tensor &a, double b) { return pg::div(b, a); })
      .def("__rtruediv__",
           [](const double a, const Tensor &b) { return pg::div(b, a); })
      .def("__neg__", [](const Tensor &a) { return pg::neg(a); })
      .def("__matmul__",
           [](const Tensor &a, const Tensor &b) { return pg::matmul(a, b); })
      .BIND_BINOP_WITH_OVERLOAD_CLASS(pow, pow)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__pow__, pow)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__rpow__, pow)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(eq, eq)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__eq__, eq)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(neq, neq)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__ne__, neq)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(gt, gt)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__gt__, gt)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(lt, lt)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__lt__, lt)
      .def("__repr__", [](const Tensor &t) { return t.str(); })
      .def(
          "__getitem__",
          [](const Tensor &arr, const py::tuple slices) {
            std::vector<hl_select_t> parsed = pybind_utils::parse_pybind_slices(
                slices, arr.shape(), arr.device());
            return pg::select(arr, parsed);
          },
          py::arg("slices").noconvert())
      .def(
          "__getitem__",
          [](const Tensor &arr, const pybind_utils::pybind_slice_item_t sl) {
            auto _tuple = py::make_tuple(sl);
            std::vector<hl_select_t> parsed = pybind_utils::parse_pybind_slices(
                _tuple, arr.shape(), arr.device());
            return pg::select(arr, parsed);
          },
          py::arg("item").noconvert());
};