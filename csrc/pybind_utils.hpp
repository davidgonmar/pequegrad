#pragma once
#include "shape.hpp"
#include "slicing.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
#include <vector>

namespace pybind_utils {
namespace py = pybind11;

using pybind_slice_item_t = std::variant<py::slice, int, std::vector<int>>;

slice_item_t parse_pybind_slice_item(const pybind_slice_item_t &item,
                                     const int shape_in_dim);
slice_t parse_pybind_slices(const py::tuple &slices, const shape_t &arr_shape);

} // namespace pybind_utils
