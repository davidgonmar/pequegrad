#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "compiler/compile.hpp"
#include "dtype.hpp"
#include "graph.hpp"
#include "npybind_utils.hpp"
#include "ops.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include <cuda.h>
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
      .value("float64", DType::Float64)
      .value("float16", DType::Float16);
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

  // device class
  py::class_<device::Device, std::shared_ptr<device::Device>>(m, "Device")
      .def("allocate", &device::Device::allocate)
      .def("memset", &device::Device::memset)
      .def("str", &device::Device::str)
      .def("__repr__", &device::Device::str);

  // tensor uitls
  m.def("tensor_precompute_again", &tensor_precompute_again);
  // device utils
  m.def("default_device", &device::get_default_device);
  m.def("device_to_string", &device::device_to_string);
  m.def("from_str", &device::from_str);
  m.def("is_cuda", &device::is_cuda);
  m.def("is_cpu", &device::is_cpu);
  m.def("force_emulated_devices", &device::force_emulated_devices);
  m.def("get_available_devices", &device::get_available_devices);
  m.def("log", &pg::log);
  m.def("exp", &pg::exp);
  m.def("cos", &pg::cos);
  m.def("sin", &pg::sin);
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
  m.def("bilinear_resize", &bilinear_resize);
  m.def("one_hot", &one_hot);

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
  m.def("as_contiguous", &as_contiguous);

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
  m.def("compile", [](std::vector<Tensor> &outs,
                      std::map<std::string, bool> options,
                      std::vector<std::tuple<py::function, py::function, bool>>
                          pattern_database) {
    // transform pattern_database to cpp map
    std::vector<std::tuple<std::function<std::vector<Tensor>(Tensor &)>,
                           std::function<Tensor(std::vector<Tensor> &)>, bool>>
        cpp_pattern_database;
    for (auto &tup : pattern_database) {
      auto matcher = [tup](Tensor &t) {
        return std::get<0>(tup)(t).cast<std::vector<Tensor>>();
      };
      auto converter = [tup](std::vector<Tensor> &t) {
        return std::get<1>(tup)(t).cast<Tensor>();
      };
      cpp_pattern_database.push_back({matcher, converter, std::get<2>(tup)});
    }
    compile(outs, options, cpp_pattern_database);
  });
  m.def("sync_cuda_device", []() { cudaDeviceSynchronize(); });
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

  class PyADPrimitive : public ADPrimitive {
  public:
    using ADPrimitive::ADPrimitive; // Inherit constructors

    virtual bool eager() override {
      PYBIND11_OVERRIDE(
          bool,        // Return type
          ADPrimitive, // Parent class
          eager        // Name of the method in C++ (must match Python name)
      );
    }

    virtual void dispatch_cpu(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) override {
      PYBIND11_OVERRIDE(
          void,           // Return type
          ADPrimitive,    // Parent class
          dispatch_cpu,   // Name of the method in C++ (must match Python name)
          inputs, outputs // Arguments
      );
    }

    virtual void dispatch_cuda(const std::vector<Tensor> &inputs,
                               std::vector<Tensor> &outputs) override {
      PYBIND11_OVERRIDE(
          void,           // Return type
          ADPrimitive,    // Parent class
          dispatch_cuda,  // Name of the method in C++ (must match Python name)
          inputs, outputs // Arguments
      );
    }

    virtual std::vector<Tensor>
    backward(const std::vector<Tensor> &primals,
             const std::vector<Tensor> &tangents,
             const std::vector<Tensor> &outputs) override {
      PYBIND11_OVERRIDE(
          std::vector<Tensor>, // Return type
          ADPrimitive,         // Parent class
          backward, // Name of the method in C++ (must match Python name)
          primals, tangents, outputs // Arguments
      );
    }

    virtual std::vector<View>
    precompute(const std::vector<Tensor> &inputs) override {
      PYBIND11_OVERRIDE(
          std::vector<View>, // Return type
          ADPrimitive,       // Parent class
          precompute, // Name of the method in C++ (must match Python name)
          inputs      // Arguments
      );
    }

    virtual std::string str() override {
      PYBIND11_OVERRIDE(
          std::string, // Return type
          ADPrimitive, // Parent class
          str          // Name of the method in C++ (must match Python name)
      );
    }
  };
  py::class_<ADPrimitive, PyADPrimitive, std::shared_ptr<ADPrimitive>>(
      m, "PyADPrimitive")
      .def(py::init([]() {
        return std::make_shared<PyADPrimitive>();
      })) // Constructor for multiple inheritance
      .def("eager", &ADPrimitive::eager)
      .def("dispatch_cpu", &ADPrimitive::dispatch_cpu)
      .def("dispatch_cuda", &ADPrimitive::dispatch_cuda)
      .def("backward", &ADPrimitive::backward)
      .def("precompute", &ADPrimitive::precompute)
      .def("str", &ADPrimitive::str)
      .def("__copy__",
           [](const ADPrimitive &self) { return ADPrimitive(self); })
      .def("__deepcopy__",
           [](const ADPrimitive &self, py::dict) { return ADPrimitive(self); });
  py::class_<View>(m, "View").def(py::init<>());

  py::class_<ADNode>(m, "ADNode")
      .def(py::init<std::shared_ptr<ADPrimitive>, std::vector<Tensor>>())
      .def("primitive",
           [](ADNode &node) {
             std::cout << "primitive\n";
             return node.primitive();
           })
      .def("children", &ADNode::children)
      .def("set_children", &ADNode::set_children)
      .def("set_primitive", [](ADNode &node, std::shared_ptr<ADPrimitive> &p) {
        node.set_primitive(p);
      });

  class PyCustomPrimitiveFromFn {
  public:
    std::function<std::vector<Tensor>(const std::vector<Tensor> &inputs)>
        basefn;
    std::optional<std::function<std::vector<Tensor>(
        const std::vector<Tensor> &primals, const std::vector<Tensor> &tangents,
        const std::vector<Tensor> &outputs)>>
        vjpfn;
    std::optional<
        std::function<std::vector<Tensor>(const std::vector<Tensor> &inputs)>>
        precomputefn;
    void set_vjpfn(py::function vjpfn) {
      this->vjpfn = [vjpfn](const std::vector<Tensor> &primals,
                            const std::vector<Tensor> &tangents,
                            const std::vector<Tensor> &outputs) {
        return vjpfn(primals, tangents, outputs).cast<std::vector<Tensor>>();
      };
    }
    void set_precomputefn(py::function precomputefn) {
      this->precomputefn = [precomputefn](const std::vector<Tensor> &inputs) {
        return precomputefn(inputs).cast<std::vector<Tensor>>();
      };
    }
    // constructor
    PyCustomPrimitiveFromFn(py::function basefn) {
      this->basefn = [basefn](const std::vector<Tensor> &inputs) {
        py::tuple args(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
          args[i] = inputs[i];
        }
        auto res = basefn(*args);
        // assert res returned a tuple py::tuple
        std::vector<Tensor> out;
        PG_CHECK_RUNTIME(py::isinstance<py::tuple>(res) ||
                             py::isinstance<py::list>(res),
                         "Custom primitive must return a tuple or list, got: " +
                             std::string(py::str(res)));
        auto castedres = res.cast<py::tuple>();
        for (size_t i = 0; i < castedres.size(); ++i) {
          out.push_back(castedres[i].cast<Tensor>());
        }
        return out;
      };
    }
  };

  py::class_<PyCustomPrimitiveFromFn>(m, "custom_prim", py::dynamic_attr())
      .def(py::init<py::function>())
      .def("setvjp", &PyCustomPrimitiveFromFn::set_vjpfn)
      .def("setprecompute", &PyCustomPrimitiveFromFn::set_precomputefn)
      .def("__call__", [](PyCustomPrimitiveFromFn &self, py::args args) {
        std::vector<Tensor> inputs;
        for (auto arg : args) {
          inputs.push_back(arg.cast<Tensor>());
        }
        PG_CHECK_RUNTIME(self.vjpfn.has_value(),
                         "Custom primitive must have a vjp function");
        FromFunctions prim =
            FromFunctions(self.basefn, self.vjpfn.value(), self.precomputefn);
        return Tensor::from_primitive_one(std::make_shared<FromFunctions>(prim),
                                          inputs);
      });

  // Custom init are functions that take no tensors and return a tuple of
  // tensors. However, they might take other python args for example def
  // arange(start, end, step):
  //   return (Tensor(np.arange(start, end, step)),)
  // so the objective is to transfer control to python to create the tensors
  class PyCustomInitFromFn {
  public:
    std::string name;
    std::function<std::vector<Tensor>(py::args)> basefn;
    // constructor
    PyCustomInitFromFn(py::function basefn) {
      this->name = basefn.attr("__name__").cast<std::string>();
      this->basefn = [basefn](py::args args) {
        auto res = basefn(*args);
        // assert res returned a tuple py::tuple
        std::vector<Tensor> out;
        PG_CHECK_RUNTIME(py::isinstance<py::tuple>(res) ||
                             py::isinstance<py::list>(res),
                         "Custom init must return a tuple or list, got: " +
                             std::string(py::str(res)));
        auto castedres = res.cast<py::tuple>();
        for (size_t i = 0; i < castedres.size(); ++i) {
          out.push_back(castedres[i].cast<Tensor>());
        }
        return out;
      };
    }

    std::vector<Tensor> compute(py::args args) {
      // capture args
      auto captured_args = args;
      // Now the thing is that we want it to be a Primitive in the computation
      // graph so do the FromFunctions thing
      FromFunctions prim = FromFunctions(
          [this, captured_args](const std::vector<Tensor> &inputs)
              -> std::vector<Tensor> { return this->basefn(captured_args); },
          [](const std::vector<Tensor> &primals,
             const std::vector<Tensor> &tangents,
             const std::vector<Tensor> &outputs) -> std::vector<Tensor> {
            throw std::runtime_error("Custom init does not support backward");
          },
          std::nullopt, this->name);
      auto x = Tensor::from_primitive_multiple(
          std::make_shared<FromFunctions>(prim), {}, device::from_str("cpu"));
      return x;
    }
  };

  py::class_<PyCustomInitFromFn>(m, "custom_init", py::dynamic_attr())
      .def(py::init<py::function>())
      .def("__call__",
           [](PyCustomInitFromFn &self,
              py::args args) -> std::variant<Tensor, std::vector<Tensor>> {
             auto a = self.compute(args);
             if (a.size() == 1) {
               return a[0];
             } else {
               return a;
             }
           });

  // module classes
  py::class_<Tensor>(m, "Tensor")
      .def_property_readonly("ndim", &Tensor::ndim)
      .def("to_", [](Tensor &t, std::shared_ptr<device::Device> device) {
        t.to_(device);
      })
      .def("to_", [](Tensor &t, std::string device) {
        t.to_(device::from_str(device));
      })
      .def("assign", &Tensor::assign)
      .def("detach", &Tensor::detach)
      .def("detach_", &Tensor::detach_)
      .def("children", &Tensor::children)
      .def("primitive", [](Tensor &t) { return t.ad_node()->primitive(); })
      .def("_inplace_as_copy", &Tensor::_inplace_as_copy)
      .def("siblings",
           [](Tensor &t) {
             std::vector<Tensor> sibs = t.ad_node()->siblings();
             return sibs;
           })
      .def("set_primitive",
           [](Tensor &t, std::shared_ptr<ADPrimitive> &p) {
             t.ad_node()->set_primitive(p);
           })
      .def("ad_context",
           [](const Tensor &t) { return t.ad_node()->primitive()->str(); })
      .def("to",
           [](Tensor &t, device::DeviceKind device) { return t.to(device::from_kind(device)); })
      .def("to",
           [](Tensor &t, std::string device) {
             auto _device = device::from_str(device);
              return t.to(_device);
           })
      .def("to", [](Tensor &t, std::shared_ptr<device::Device> device) {
        return t.to(device);
      })
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return Tensor::from_numpy(np_array);
           })
      .def("unsqueeze",
           [](const Tensor &a, py::object axes) {
             if (py::isinstance<py::int_>(axes)) {
               return unsqueeze(a, axes.cast<axis_t>());
             } else if (py::isinstance<py::list>(axes) ||
                        py::isinstance<py::tuple>(axes)) {
               return unsqueeze(a, axes.cast<axes_t>());
             } else {
               throw std::runtime_error(
                   "unsqueeze: axes must be an int, list or "
                   "tuple");
             }
           })
      .def("squeeze",
           [](const Tensor &a, py::object axes) {
             if (py::isinstance<py::int_>(axes)) {
               return squeeze(a, axes.cast<axis_t>());
             } else if (py::isinstance<py::list>(axes) ||
                        py::isinstance<py::tuple>(axes)) {
               return squeeze(a, axes.cast<axes_t>());
             } else {
               throw std::runtime_error("squeeze: axes must be an int, list or "
                                        "tuple");
             }
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
      .def("astype", [](Tensor &t, DType dtype) { return t.astype(dtype); })
      .def("astype",
           [](Tensor &t, std::string dtype) {
             return t.astype(dtype_from_string(dtype));
           })
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
      .def("replace_child",
           [](Tensor &t, const Tensor &old_child, const Tensor &new_child) {
             t.ad_node()->replace_child(old_child, new_child);
           })
      .def("is_evaled", &Tensor::is_evaled)
      .def("__hash__", [](const Tensor &t) { return t.id; })
      .def_property_readonly("id", [](const Tensor &t) { return t.id; })
      .def_property_readonly(
          "position", [](const Tensor &t) { return t.ad_node()->position(); })
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
      .def_property_readonly("strides",
                             [](const Tensor &t) { return t.strides(); })
      .def_property_readonly("dtype", [](const Tensor &t) { return t.dtype(); })
      .def_property_readonly("device",
                             [](const Tensor &t) { return t.device(); })
      .def(py::init([](py::array_t<float> np_array, device::DeviceKind device) {
             return Tensor::from_numpy(np_array, device::from_kind(device));
           }),
           py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)
      .def(py::init([](py::array_t<int> np_array, device::DeviceKind device) {
             return Tensor::from_numpy(np_array, device::from_kind(device));
           }),
           py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)
      .def(
          py::init([](py::array_t<double> np_array, device::DeviceKind device) {
            return Tensor::from_numpy(np_array, device::from_kind(device));
          }),
          py::arg("np_array"), py::arg("device") = device::DeviceKind::CPU)
      .def(py::init([](py::array_t<float> np_array, std::string device) {
             return Tensor::from_numpy(np_array, device::from_str(device));
           }),
           py::arg("np_array"), py::arg("device") = "cpu")
      .def(py::init([](py::array_t<int> np_array, std::string device) {
             return Tensor::from_numpy(np_array, device::from_str(device));
           }),
           py::arg("np_array"), py::arg("device") = "cpu")
      .def(py::init([](py::array_t<double> np_array, std::string device) {
             return Tensor::from_numpy(np_array, device::from_str(device));
           }),
           py::arg("np_array"), py::arg("device") = "cpu")
      .def(py::init([](py::array_t<float> np_array, std::shared_ptr<device::Device> device) {
             return Tensor::from_numpy(np_array, device);
           }),
           py::arg("np_array"), py::arg("device"))
      .def(py::init([](py::array_t<int> np_array, std::shared_ptr<device::Device> device) {
             return Tensor::from_numpy(np_array, device);
           }),
            py::arg("np_array"), py::arg("device"))
      .def(py::init([](py::array_t<double> np_array, std::shared_ptr<device::Device> device) {
             return Tensor::from_numpy(np_array, device);
            }),
            py::arg("np_array"), py::arg("device"))

      .def(py::init([](Tensor &orig) { return Tensor(orig); }))
      .BIND_BINOP_WITH_OVERLOAD_CLASS(add, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__add__, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__radd__, add)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(sub, sub)
      .BIND_BINOP_WITH_OVERLOAD_CLASS(__sub__, sub)
      .def("__rsub__",
           [](const Tensor &a, const Tensor &b) { return pg::sub(b, a); })
      .def("__rsub__", [](const Tensor &a, double b) { return pg::sub(b, a); })
      .def("__rsub__",
           [](const double a, const Tensor &b) { return pg::sub(b, a); })
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
      .def("__rpow__",
           [](const Tensor &a, const Tensor &b) { return pg::pow(b, a); })
      .def("__rpow__", [](const Tensor &a, double b) { return pg::pow(b, a); })
      .def("__rpow__",
           [](const double a, const Tensor &b) { return pg::pow(b, a); })
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
          py::arg("item").noconvert())
      .def("view", [](const Tensor &t) { return t.view(); })
      .def("primitive",
           [](const Tensor &t) { return t.ad_node()->primitive(); })
      .def("ad_node", [](const Tensor &t) { return t.ad_node(); })
      .def("_eval_assume_inputs_evaled",
      [](Tensor &t) { return t._eval_assume_inputs_evaled();
      });

  // primitives
  py::class_<BroadcastTo, std::shared_ptr<BroadcastTo>>(m, "BroadcastTo")
      .def("set_shape_to", &BroadcastTo::set_shape_to)
      .def("get_created_axes", &BroadcastTo::get_created_axes)
      .def("get_broadcasted_axes", &BroadcastTo::get_broadcasted_axes)
      .def("shape_to", &BroadcastTo::shape_to);
};