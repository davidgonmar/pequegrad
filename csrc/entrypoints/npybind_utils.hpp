#pragma once
#include "ops.hpp"
#include "shape.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
#include <vector>

namespace pg {
namespace pybind_utils {
namespace py = pybind11;

using pybind_slice_item_t =
    std::variant<py::slice, int, Tensor, py::array_t<int>>;

hl_select_t parse_pybind_slice_item(const pybind_slice_item_t &item,
                                    const int shape_in_dim);
std::vector<hl_select_t>
parse_pybind_slices(const py::tuple &slices, const shape_t &arr_shape,
                    std::shared_ptr<device::Device> inp_device);

} // namespace pybind_utils
} // namespace pg