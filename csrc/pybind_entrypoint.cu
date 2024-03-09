#include "cuda_array/cuda_array.cuh"
#include "kernels/all.cuh"
#include "utils.cuh"
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

PYBIND11_MODULE(pequegrad_cu, m) {
  namespace py = pybind11;

  m.attr("__device_name__") = "cuda";

  py::class_<CudaArray>(m, "CudaArray")
      .def_readonly("shape", &CudaArray::shape)
      .def_readonly("strides", &CudaArray::strides)
      .def("clone", &CudaArray::clone)
      .def("broadcast_to", &CudaArray::broadcast_to)
      .def("to_numpy",
           [](const CudaArray &arr) -> NpArrayVariant {
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
             return CudaArray::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<int> np_array) {
             return CudaArray::from_numpy(np_array);
           })
      .def("from_numpy",
           [](py::array_t<double> np_array) {
             return CudaArray::from_numpy(np_array);
           })
      .def("__repr__", [](const CudaArray &arr) { return arr.to_string(); })
      .def(
          "add",
          [](const CudaArray &arr, const CudaArray &other) {
            return arr.binop(other, BinaryKernelType::ADD);
          },
          py::arg("other").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, const py::array_t<int> np_array) {
            return arr.binop(np_array, BinaryKernelType::ADD);
          },
          py::arg("np_array").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, const py::array_t<float> np_array) {
            return arr.binop(np_array, BinaryKernelType::ADD);
          },
          py::arg("np_array").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, const py::array_t<double> np_array) {
            return arr.binop(np_array, BinaryKernelType::ADD);
          },
          py::arg("np_array").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, int scalar) {
            return arr.binop(scalar, BinaryKernelType::ADD);
          },
          py::arg("scalar").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, float scalar) {
            return arr.binop(scalar, BinaryKernelType::ADD);
          },
          py::arg("scalar").noconvert())
      .def(
          "add",
          [](const CudaArray &arr, double scalar) {
            return arr.binop(scalar, BinaryKernelType::ADD);
          },
          py::arg("scalar").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, const CudaArray &other) {
            return arr.binop(other, BinaryKernelType::SUB);
          },
          py::arg("other").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, const py::array_t<int> np_array) {
            return arr.binop(np_array, BinaryKernelType::SUB);
          },
          py::arg("np_array").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, const py::array_t<float> np_array) {
            return arr.binop(np_array, BinaryKernelType::SUB);
          },
          py::arg("np_array").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, const py::array_t<double> np_array) {
            return arr.binop(np_array, BinaryKernelType::SUB);
          },
          py::arg("np_array").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, int scalar) {
            return arr.binop(scalar, BinaryKernelType::SUB);
          },
          py::arg("scalar").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, float scalar) {
            return arr.binop(scalar, BinaryKernelType::SUB);
          },
          py::arg("scalar").noconvert())
      .def(
          "sub",
          [](const CudaArray &arr, double scalar) {
            return arr.binop(scalar, BinaryKernelType::SUB);
          },
          py::arg("scalar").noconvert())
      .def("mul",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::MULT);
           })
      .def(
          "mul",
          [](const CudaArray &arr, const py::array_t<float> np_array) {
            return arr.binop(np_array, BinaryKernelType::MULT);
          },
          py::arg("np_array").noconvert())
      .def(
          "mul",
          [](const CudaArray &arr, const py::array_t<int> np_array) {
            return arr.binop(np_array, BinaryKernelType::MULT);
          },
          py::arg("np_array").noconvert())
      .def(
          "mul",
          [](const CudaArray &arr, const py::array_t<double> np_array) {
            return arr.binop(np_array, BinaryKernelType::MULT);
          },
          py::arg("np_array").noconvert())
      .def(
          "mul",
          [](const CudaArray &arr, const double scalar) {
            return arr.binop(scalar, BinaryKernelType::MULT);
          },
          py::arg("scalar").noconvert())
      .def(
          "mul",
          [](const CudaArray &arr, const float scalar) {
            return arr.binop(scalar, BinaryKernelType::MULT);
          },
          py::arg("scalar").noconvert())
      .def(
          "mul",
          [](const CudaArray &arr, const int scalar) {
            return arr.binop(scalar, BinaryKernelType::MULT);
          },
          py::arg("scalar").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const CudaArray &other) {
            return arr.binop(other, BinaryKernelType::DIV);
          },
          py::arg("other").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const py::array_t<int> np_array) {
            return arr.binop(np_array, BinaryKernelType::DIV);
          },
          py::arg("np_array").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const py::array_t<float> np_array) {
            return arr.binop(np_array, BinaryKernelType::DIV);
          },
          py::arg("np_array").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const py::array_t<double> np_array) {
            return arr.binop(np_array, BinaryKernelType::DIV);
          },
          py::arg("np_array").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const int scalar) {
            return arr.binop(scalar, BinaryKernelType::DIV);
          },
          py::arg("scalar").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const float scalar) {
            return arr.binop(scalar, BinaryKernelType::DIV);
          },
          py::arg("scalar").noconvert())
      .def(
          "div",
          [](const CudaArray &arr, const double scalar) {
            return arr.binop(scalar, BinaryKernelType::DIV);
          },
          py::arg("scalar").noconvert())
      .def("el_wise_max",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::ELEMENT_WISE_MAX);
           })
      .def("pow",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::POW);
           })
      .def("pow",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::POW);
           })
      .def("pow",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::POW);
           })
      .def("eq",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::EQUAL);
           })
      .def("ne",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::NOT_EQUAL);
           })
      .def("lt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::LESS);
           })
      .def("le",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::LESS_EQUAL);
           })
      .def("gt",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::GREATER);
           })
      .def("ge",
           [](const CudaArray &arr, const CudaArray &other) {
             return arr.binop(other, BinaryKernelType::GREATER_EQUAL);
           })
      .def("contiguous", &CudaArray::as_contiguous)
      .def("exp",
           [](const CudaArray &arr) {
             return arr.elwiseop(UnaryKernelType::EXP);
           })
      .def("log",
           [](const CudaArray &arr) {
             return arr.elwiseop(UnaryKernelType::LOG);
           })
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
             return cond.ternaryop(a, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<float> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<float> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<int> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<int> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<int> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<double> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<double> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<int> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<int> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<float> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const py::array_t<float> np_array,
              const CudaArray &b) {
             return a.ternaryop(np_array, b, TernaryKernelType::WHERE);
           })
      .def("where",
           [](const CudaArray &a, const CudaArray &cond,
              const py::array_t<double> np_array) {
             return cond.ternaryop(a, np_array, TernaryKernelType::WHERE);
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
      .def("squeeze",
           [](const CudaArray &arr, axes_t axes) { return arr.squeeze(axes); })
      .def("squeeze", [](const CudaArray &arr) { return arr.squeeze(); })
      .def("unsqueeze", [](const CudaArray &arr,
                           axis_t axis) { return arr.unsqueeze(axis); })
      .def("unsqueeze", [](const CudaArray &arr,
                           axes_t axes) { return arr.unsqueeze(axes); })
      .def("reshape",
           [](const CudaArray &arr, std::vector<int> new_shape) {
             return arr.reshape(new_shape);
           })
      .def("im2col",
           [](const CudaArray &arr, shape_t kernel_shape, shape_t stride,
              shape_t dilation) {
             if (stride.size() != 2) {
               throw std::runtime_error("Stride must be a 2D array");
             }
             if (dilation.size() != 2) {
               throw std::runtime_error("Dilation must be a 2D array");
             }
             return arr.im2col(kernel_shape, stride.at(0), stride.at(1),
                               dilation.at(0), dilation.at(1));
           })
      .def("col2im",
           [](const CudaArray &arr, shape_t kernel_shape, shape_t output_shape,
              shape_t stride, shape_t dilation) {
             if (stride.size() != 2) {
               throw std::runtime_error("Stride must be a 2D array");
             }
             if (dilation.size() != 2) {
               throw std::runtime_error("Dilation must be a 2D array");
             }
             return arr.col2im(kernel_shape, output_shape, stride.at(0),
                               stride.at(1), dilation.at(0), dilation.at(1));
           })
      .def_static(
          "fill",
          [](shape_t shape, py::handle value, const std::string &dtype) {
            if (dtype == "float32") {
              return CudaArray::fill<float>(shape, value.cast<float>());
            } else if (dtype == "int32") {
              return CudaArray::fill<int>(shape, value.cast<int>());
            } else if (dtype == "float64") {
              return CudaArray::fill<double>(shape, value.cast<double>());
            } else {
              throw std::runtime_error("Unsupported data type");
            }
          })
      .def("__getitem__",
           [](const CudaArray &arr, shape_t index) -> ItemVariant {
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
      .def("dtype",
           [](const CudaArray &arr) -> py::dtype {
             switch (arr.dtype) {
             case DType::Float32:
               return py::dtype::of<float>();
             case DType::Int32:
               return py::dtype::of<int>();
             case DType::Float64:
               return py::dtype::of<double>();
             default:
               throw std::runtime_error("Unsupported data type");
             }
           })
      .def("slice", [](const CudaArray &arr,
                       std::variant<std::vector<std::variant<py::slice, int>>,
                                    py::slice, int>
                           slices) {
        slice_t slice_pairs;

        if (std::holds_alternative<py::slice>(slices)) {
          py::slice single_slice = std::get<py::slice>(slices);
          py::ssize_t start, stop, step, slicelength;
          if (single_slice.compute(arr.size, &start, &stop, &step,
                                   &slicelength)) {
            slice_pairs.push_back(std::make_pair(static_cast<int>(start),
                                                 static_cast<int>(stop)));
          }
        } else if (std::holds_alternative<int>(slices)) {
          slice_pairs.push_back(std::get<int>(slices));
        } else {
          std::vector<std::variant<py::slice, int>> slices_vector =
              std::get<std::vector<std::variant<py::slice, int>>>(slices);
          for (auto slice : slices_vector) {
            if (std::holds_alternative<py::slice>(slice)) {
              py::slice single_slice = std::get<py::slice>(slice);
              py::ssize_t start, stop, step, slicelength;
              if (single_slice.compute(arr.size, &start, &stop, &step,
                                       &slicelength)) {
                slice_pairs.push_back(std::make_pair(static_cast<int>(start),
                                                     static_cast<int>(stop)));
              }
            } else if (std::holds_alternative<int>(slice)) {
              slice_pairs.push_back(std::get<int>(slice));
            }
          }
        }
        return arr.slice(slice_pairs);
      });
}