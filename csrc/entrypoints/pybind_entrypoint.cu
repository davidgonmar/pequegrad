#include "cuda_tensor/cuda_tensor.cuh"
#include "pybind_utils.cuh"
#include "cuda_tensor/cuda_utils.cuh"
#include <cuda_runtime.h>
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

#define REGISTER_BINARY_OP_ALL(PY_FN_NAME, KERNEL_NAME)                        \
  .def(                                                                        \
      PY_FN_NAME,                                                              \
      [](const CudaTensor &arr, const CudaTensor &other) {                     \
        return arr.binop(other, BinaryKernelType::KERNEL_NAME);                \
      },                                                                       \
      py::arg("other").noconvert())                                            \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, const py::array_t<int> np_array) {         \
            return arr.binop(np_array, BinaryKernelType::KERNEL_NAME);         \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, const py::array_t<float> np_array) {       \
            return arr.binop(np_array, BinaryKernelType::KERNEL_NAME);         \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, const py::array_t<double> np_array) {      \
            return arr.binop(np_array, BinaryKernelType::KERNEL_NAME);         \
          },                                                                   \
          py::arg("np_array").noconvert())                                     \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, int scalar) {                              \
            return arr.binop(scalar, BinaryKernelType::KERNEL_NAME);           \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, float scalar) {                            \
            return arr.binop(scalar, BinaryKernelType::KERNEL_NAME);           \
          },                                                                   \
          py::arg("scalar").noconvert())                                       \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &arr, double scalar) {                           \
            return arr.binop(scalar, BinaryKernelType::KERNEL_NAME);           \
          },                                                                   \
          py::arg("scalar").noconvert())

#define REGISTER_TERMARY_OP_ALL(PY_FN_NAME, KERNEL_NAME)                       \
  .def(                                                                        \
      PY_FN_NAME,                                                              \
      [](const CudaTensor &cond, const CudaTensor &a, const CudaTensor &b) {   \
        return cond.ternaryop(a, b, TernaryKernelType::KERNEL_NAME);           \
      },                                                                       \
      py::arg("a").noconvert(), py::arg("b").noconvert())                      \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a,                      \
             const py::array_t<int> np_array) {                                \
            return cond.ternaryop(a, np_array,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("np_array").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a,                      \
             const py::array_t<float> np_array) {                              \
            return cond.ternaryop(a, np_array,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("np_array").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a,                      \
             const py::array_t<double> np_array) {                             \
            return cond.ternaryop(a, np_array,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("np_array").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<int> np_array,          \
             const CudaTensor &b) {                                            \
            return cond.ternaryop(np_array, b,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("b").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<float> np_array,        \
             const CudaTensor &b) {                                            \
            return cond.ternaryop(np_array, b,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("b").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<double> np_array,       \
             const CudaTensor &b) {                                            \
            return cond.ternaryop(np_array, b,                                 \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("b").noconvert())           \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<int> np_array,          \
             const py::array_t<int> np_array2) {                               \
            return cond.ternaryop(np_array, np_array2,                         \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("np_array2").noconvert())   \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<float> np_array,        \
             const py::array_t<float> np_array2) {                             \
            return cond.ternaryop(np_array, np_array2,                         \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("np_array2").noconvert())   \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const py::array_t<double> np_array,       \
             const py::array_t<double> np_array2) {                            \
            return cond.ternaryop(np_array, np_array2,                         \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("np_array").noconvert(), py::arg("np_array2").noconvert())   \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, int scalar, int scalar2) {                \
            return cond.ternaryop(scalar, scalar2,                             \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("scalar2").noconvert())       \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, float scalar, float scalar2) {            \
            return cond.ternaryop(scalar, scalar2,                             \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("scalar2").noconvert())       \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, double scalar, double scalar2) {          \
            return cond.ternaryop(scalar, scalar2,                             \
                                  TernaryKernelType::KERNEL_NAME);             \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("scalar2").noconvert())       \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, int scalar, const CudaTensor &b) {        \
            return cond.ternaryop(scalar, b, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("b").noconvert())             \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, float scalar, const CudaTensor &b) {      \
            return cond.ternaryop(scalar, b, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("b").noconvert())             \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, double scalar, const CudaTensor &b) {     \
            return cond.ternaryop(scalar, b, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("scalar").noconvert(), py::arg("b").noconvert())             \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a, int scalar) {        \
            return cond.ternaryop(a, scalar, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("scalar").noconvert())             \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a, float scalar) {      \
            return cond.ternaryop(a, scalar, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("scalar").noconvert())             \
      .def(                                                                    \
          PY_FN_NAME,                                                          \
          [](const CudaTensor &cond, const CudaTensor &a, double scalar) {     \
            return cond.ternaryop(a, scalar, TernaryKernelType::KERNEL_NAME);  \
          },                                                                   \
          py::arg("a").noconvert(), py::arg("scalar").noconvert())

#define REGISTER_REDUCER_ALL(PY_FN_NAME, CUDA_ARR_METHOD_NAME)                 \
  .def(                                                                        \
      PY_FN_NAME,                                                              \
      [](const CudaTensor &arr, py::object axis, bool keepdims) {              \
        if (axis.is_none()) {                                                  \
          return arr.CUDA_ARR_METHOD_NAME(keepdims);                           \
        } else if (py::isinstance<py::int_>(axis)) {                           \
          return arr.CUDA_ARR_METHOD_NAME(axis.cast<int>(), keepdims);         \
        } else if (py::isinstance<py::list>(axis)) {                           \
          return arr.CUDA_ARR_METHOD_NAME(axis.cast<axes_t>(), keepdims);      \
        } else if (py::isinstance<py::tuple>(axis)) {                          \
          return arr.CUDA_ARR_METHOD_NAME(axis.cast<axes_t>(), keepdims);      \
        } else {                                                               \
          throw std::runtime_error("Invalid axis");                            \
        }                                                                      \
      },                                                                       \
      py::arg("axis"), py::arg("keepdims") = false)

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;

  m.attr("__device_name__") = "cuda";

  py::class_<CudaTensor>(m, "CudaTensor")
      .def_property_readonly("shape",
                             [](const CudaTensor &arr) {
                               py::tuple shape(arr.shape.size());
                               for (size_t i = 0; i < arr.shape.size(); i++) {
                                 shape[i] = arr.shape[i];
                               }
                               return shape;
                             })
      .def_property_readonly("strides",
                             [](const CudaTensor &arr) {
                               py::tuple strides(arr.strides.size());
                               for (size_t i = 0; i < arr.strides.size(); i++) {
                                 strides[i] = arr.strides[i];
                               }
                               return strides;
                             })
      .def("clone", &CudaTensor::clone)
      .def("broadcast_to", &CudaTensor::broadcast_to)
      .def("to_numpy",
           [](const CudaTensor &arr) -> NpArrayVariant {
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
      .def("from_numpy",
           [](py::array_t<float> np_array) {
             return CudaTensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<int> np_array) {
             return CudaTensor::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<double> np_array) {
             return CudaTensor::from_numpy(np_array);
           })
 REGISTER_BINARY_OP_ALL("add",
                                     ADD) REGISTER_BINARY_OP_ALL("__add__", ADD)
          REGISTER_BINARY_OP_ALL("__radd__", ADD) REGISTER_BINARY_OP_ALL(
              "sub",
              SUB) REGISTER_BINARY_OP_ALL("__sub__",
                                          SUB) REGISTER_BINARY_OP_ALL("__rsub_"
                                                                      "_",
                                                                      SUB)
              REGISTER_BINARY_OP_ALL("mul", MULT) REGISTER_BINARY_OP_ALL(
                  "__mul__",
                  MULT) REGISTER_BINARY_OP_ALL("__rmul__",
                                               MULT) REGISTER_BINARY_OP_ALL("di"
                                                                            "v",
                                                                            DIV)
                  REGISTER_BINARY_OP_ALL("__truediv__", DIV) REGISTER_BINARY_OP_ALL(
                      "__rtruediv__",
                      DIV) REGISTER_BINARY_OP_ALL("el_wise_max",
                                                  ELEMENT_WISE_MAX)
                      REGISTER_BINARY_OP_ALL("pow", POW) REGISTER_BINARY_OP_ALL(
                          "__pow__", POW) REGISTER_BINARY_OP_ALL("__rpow__",
                                                                 POW)
                          REGISTER_BINARY_OP_ALL(
                              "eq", EQUAL) REGISTER_BINARY_OP_ALL("__eq__",
                                                                  EQUAL)
                              REGISTER_BINARY_OP_ALL(
                                  "ne",
                                  NOT_EQUAL) REGISTER_BINARY_OP_ALL("__ne__",
                                                                    NOT_EQUAL)
                                  REGISTER_BINARY_OP_ALL(
                                      "lt",
                                      LESS) REGISTER_BINARY_OP_ALL("__lt__",
                                                                   LESS)
                                      REGISTER_BINARY_OP_ALL("le", LESS_EQUAL)
                                          REGISTER_BINARY_OP_ALL("__le__",
                                                                 LESS_EQUAL)
                                              REGISTER_BINARY_OP_ALL("gt",
                                                                     GREATER)
                                                  REGISTER_BINARY_OP_ALL(
                                                      "__gt__", GREATER)
                                                      REGISTER_BINARY_OP_ALL(
                                                          "ge", GREATER_EQUAL)
                                                          REGISTER_BINARY_OP_ALL(
                                                              "__ge__",
                                                              GREATER_EQUAL)
      .def("contiguous", &CudaTensor::as_contiguous)
      .def("exp",
           [](const CudaTensor &arr) {
             return arr.elwiseop(UnaryKernelType::EXP);
           })
      .def("log",
           [](const CudaTensor &arr) {
             return arr.elwiseop(UnaryKernelType::LOG);
           })
      .def("permute",
           [](const CudaTensor &arr, py::args axes) {
             shape_t axes_vec = axes.cast<shape_t>();
             return arr.permute(axes_vec);
           })
      .def("transpose",
           [](const CudaTensor &arr) {
             if (arr.ndim() < 2) {
               return arr;
             }
             shape_t axes(arr.ndim());
             std::iota(axes.begin(), axes.end(), 0);
             std::swap(axes[0], axes[1]);
             return arr.permute(axes);
           })
      .def(
          "swapaxes",
          [](const CudaTensor &arr, int axis1, int axis2) {
            // we need to allow for negative indexing, but permute will be
            // passed the actual axis
            shape_t axes(arr.ndim());
            for (int i = 0; i < arr.ndim(); i++) {
              axes[i] = i;
            }
            // negative indexing is allowed
            if (axis1 < 0) {
              axis1 = arr.ndim() + axis1;
            }
            if (axis2 < 0) {
              axis2 = arr.ndim() + axis2;
            }
            std::swap(axes[axis1], axes[axis2]);
            return arr.permute(axes);
          },
          py::arg("axis1"), py::arg("axis2"))
      .def("is_contiguous", &CudaTensor::is_contiguous)
      .def("matmul", [](const CudaTensor &arr,
                        const CudaTensor &other) { return arr.mat_mul(other); })
      .def("outer_product",
           [](const CudaTensor &arr, const CudaTensor &other) {
             return arr.outer_product(other);
           }) REGISTER_TERMARY_OP_ALL("where", WHERE)
      .def_static(
          "where_static",
          [](const CudaTensor &cond, const py::handle a, const py::handle b) {
            if (py::isinstance<CudaTensor>(a) &&
                py::isinstance<CudaTensor>(b)) {
              return cond.ternaryop(a.cast<CudaTensor>(), b.cast<CudaTensor>(),
                                    TernaryKernelType::WHERE);
            } else if (py::isinstance<py::array_t<float>>(a) &&
                       py::isinstance<py::array_t<float>>(b)) {
              return cond.ternaryop(a.cast<py::array_t<float>>(),
                                    b.cast<py::array_t<float>>(),
                                    TernaryKernelType::WHERE);
            } else if (py::isinstance<py::array_t<int>>(a) &&
                       py::isinstance<py::array_t<int>>(b)) {
              return cond.ternaryop(a.cast<py::array_t<int>>(),
                                    b.cast<py::array_t<int>>(),
                                    TernaryKernelType::WHERE);
            } else if (py::isinstance<py::array_t<double>>(a) &&
                       py::isinstance<py::array_t<double>>(b)) {
              return cond.ternaryop(a.cast<py::array_t<double>>(),
                                    b.cast<py::array_t<double>>(),
                                    TernaryKernelType::WHERE);
            } else { // for scalars, we must create a new instance of the array,
                     // since ternaryop doesnt support scalars
              if (py::isinstance<py::int_>(a) && py::isinstance<py::int_>(b)) {
                return cond.ternaryop(
                    CudaTensor::fill<int>(cond.shape, a.cast<int>()),
                    CudaTensor::fill<int>(cond.shape, b.cast<int>()),
                    TernaryKernelType::WHERE);
              } else if (py::isinstance<py::float_>(a) &&
                         py::isinstance<py::float_>(b)) {
                return cond.ternaryop(
                    CudaTensor::fill<float>(cond.shape, a.cast<float>()),
                    CudaTensor::fill<float>(cond.shape, b.cast<float>()),
                    TernaryKernelType::WHERE);
              } else {
                throw std::runtime_error("Invalid type for a and b: " +
                                         std::string(py::str(a.get_type())) +
                                         " and " +
                                         std::string(py::str(b.get_type())));
              }
            }
          },
          py::arg("cond"), py::arg("a"), py::arg("b"))

          REGISTER_REDUCER_ALL("sum", sum) REGISTER_REDUCER_ALL("max", max)
              REGISTER_REDUCER_ALL("mean", mean)
      .def(
          "squeeze",
          [](const CudaTensor &arr, axis_t axis) { return arr.squeeze(axis); },
          py::arg("axis"))
      .def(
          "squeeze",
          [](const CudaTensor &arr, axes_t axes) { return arr.squeeze(axes); },
          py::arg("axes"))
      .def("squeeze", [](const CudaTensor &arr) { return arr.squeeze(); })
      .def("unsqueeze", [](const CudaTensor &arr,
                           axis_t axis) { return arr.unsqueeze(axis); })
      .def("unsqueeze", [](const CudaTensor &arr,
                           axes_t axes) { return arr.unsqueeze(axes); })
      .def("reshape",
           [](const CudaTensor &arr, py::args new_shape) {
             std::vector<int> shape_vec = new_shape.cast<std::vector<int>>();
             return arr.reshape(shape_vec);
           })
      .def(
          "im2col",
          [](const CudaTensor &arr, shape_t kernel_shape, py::handle _stride,
             py::handle _dilation) {
            shape_t stride;
            shape_t dilation;
            if (py::isinstance<py::int_>(_stride)) {
              size_t stride_val = _stride.cast<int>();
              stride = {stride_val, stride_val};
            } else {
              stride = _stride.cast<shape_t>();
            }
            if (py::isinstance<py::int_>(_dilation)) {
              size_t dilation_val = _dilation.cast<int>();
              dilation = {dilation_val, dilation_val};
            } else {
              dilation = _dilation.cast<shape_t>();
            }

            if (stride.size() != 2) {
              throw std::runtime_error("Stride must be a 2D array");
            }
            if (dilation.size() != 2) {
              throw std::runtime_error("Dilation must be a 2D array");
            }
            return arr.im2col(kernel_shape, stride.at(0), stride.at(1),
                              dilation.at(0), dilation.at(1));
          },
          py::arg("kernel_shape"), py::arg("stride"), py::arg("dilation"))
      .def(
          "col2im",
          [](const CudaTensor &arr, shape_t kernel_shape, shape_t output_shape,
             py::handle _stride, py::handle _dilation) {
            shape_t stride;
            shape_t dilation;
            if (py::isinstance<py::int_>(_stride)) {
              size_t stride_val = _stride.cast<int>();
              stride = {stride_val, stride_val};
            } else {
              stride = _stride.cast<shape_t>();
            }
            if (py::isinstance<py::int_>(_dilation)) {
              size_t dilation_val = _dilation.cast<int>();
              dilation = {dilation_val, dilation_val};
            } else {
              dilation = _dilation.cast<shape_t>();
            }
            if (stride.size() != 2) {
              throw std::runtime_error("Stride must be a 2D array");
            }
            if (dilation.size() != 2) {
              throw std::runtime_error("Dilation must be a 2D array");
            }
            return arr.col2im(kernel_shape, output_shape, stride.at(0),
                              stride.at(1), dilation.at(0), dilation.at(1));
          },
          py::arg("kernel_shape"), py::arg("output_shape"), py::arg("stride"),
          py::arg("dilation"))
      .def_static(
          "fill",
          [](std::variant<shape_t, size_t> sh, py::handle value,
             std::string dtype) {
            shape_t shape;
            if (std::holds_alternative<shape_t>(sh)) {
              shape = std::get<shape_t>(sh);
            } else {
              shape = {std::get<size_t>(sh)};
            }
            if (dtype == "float32") {
              if (py::isinstance<py::int_>(value)) {
                return CudaTensor::fill<float>(
                    shape, static_cast<float>(value.cast<int>()));
              } else if (py::isinstance<py::float_>(value)) {
                return CudaTensor::fill<float>(shape, value.cast<float>());
              } else {
                throw std::runtime_error(
                    "Unsupported value type for dtype float32");
              }
            } else if (dtype == "int32") {
              if (py::isinstance<py::float_>(value)) {
                return CudaTensor::fill<int>(
                    shape, static_cast<int>(value.cast<float>()));
              } else if (py::isinstance<py::int_>(value)) {
                return CudaTensor::fill<int>(shape, value.cast<int>());
              } else {
                throw std::runtime_error(
                    "Unsupported value type for dtype int32");
              }
            } else if (dtype == "float64") {
              if (py::isinstance<py::int_>(value)) {
                return CudaTensor::fill<double>(
                    shape, static_cast<double>(value.cast<int>()));
              } else if (py::isinstance<py::float_>(value)) {
                return CudaTensor::fill<double>(
                    shape, static_cast<double>(value.cast<float>()));
              } else {
                throw std::runtime_error(
                    "Unsupported value type for dtype float64");
              }
            } else {
              throw std::runtime_error("Unsupported data type");
            }
          },
          py::arg("shape"), py::arg("value"), py::arg("dtype") = "float32")
      .def("__getitem__",
           [](const CudaTensor &arr, shape_t index) -> ItemVariant {
             switch (arr.dtype) {
             case DType::Float32:
               return ItemVariant(arr.getitem<float>(index));
             case DType::Int32:
               return ItemVariant(arr.getitem<int>(index));
             case DType::Float64:
               return ItemVariant(arr.getitem<double>(index));
             default:
               throw std::runtime_error("Unsupported data type");
             }
           })
      .def_property_readonly("dtype",
                             [](const CudaTensor &arr) -> std::string {
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
      .def(
          "slice",
          [](const CudaTensor &arr, const py::tuple slices) {
            slice_t parsed =
                pybind_utils::parse_pybind_slices(slices, arr.shape);
            return arr.slice(parsed);
          },
          py::arg("slices").noconvert())
      .def(
          "slice",
          [](const CudaTensor &arr,
             const pybind_utils::pybind_slice_item_t sl) {
            auto _tuple = py::make_tuple(sl);
            slice_t parsed =
                pybind_utils::parse_pybind_slices(_tuple, arr.shape);
            return arr.slice(parsed);
          },
          py::arg("item"))
      .def("assign",
           [](CudaTensor &arr, const py::tuple slices, CudaTensor vals) {
             slice_t parsed =
                 pybind_utils::parse_pybind_slices(slices, arr.shape);

             return arr.assign(parsed, vals);
           })
      .def("assign",
           [](CudaTensor &arr, const pybind_utils::pybind_slice_item_t item,
              CudaTensor vals) {
             auto _tuple = py::make_tuple(item);
             slice_t parsed =
                 pybind_utils::parse_pybind_slices(_tuple, arr.shape);
             return arr.assign(parsed, vals);
           })
      .def(
          "expand_dims",
          [](const CudaTensor &arr, int axis) { return arr.unsqueeze(axis); },
          py::arg("axis"))
      .def(
          "expand_dims",
          [](const CudaTensor &arr, axes_t axis) {
            return arr.unsqueeze(axis);
          },
          py::arg("axis"))
      .def_property_readonly("ndim", &CudaTensor::ndim)
      .def(
          "astype",
          [](const CudaTensor &arr, std::string dtype) {
            if (dtype == "float32") {
              return arr.astype(DType::Float32);
            } else if (dtype == "int32") {
              return arr.astype(DType::Int32);
            } else if (dtype == "float64") {
              return arr.astype(DType::Float64);
            } else {
              throw std::runtime_error("Unsupported data type");
            }
          },
          py::arg("dtype") = "float32");
}