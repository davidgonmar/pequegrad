
#include "npybind_utils.hpp"
#include "ops.hpp"
#include "utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <variant>
#include <vector>
namespace pg {
namespace pybind_utils {

hl_select_t parse_pybind_slice_item(const pybind_slice_item_t &item,
                                    const int shape_in_dim) {
  if (std::holds_alternative<py::slice>(item)) {
    py::slice s = std::get<py::slice>(item);

    py::ssize_t start, stop, step, slicelength;
    if (!s.compute(shape_in_dim, &start, &stop, &step, &slicelength)) {
      throw std::runtime_error("Error during slices parsing");
    }
    return SelectWithSlice(start, stop, step);
  } else if (std::holds_alternative<int>(item)) {
    return SelectWithSingleIdx(std::get<int>(item));
  } else if (std::holds_alternative<Tensor>(item)) {
    return std::get<Tensor>(item);
  } else {
    throw std::runtime_error("[parse_pybind_slice_item] Invalid slice");
  }
}
std::vector<hl_select_t> parse_pybind_slices(const py::tuple &slices,
                                             const shape_t &arr_shape,
                                             device::DeviceKind inp_device) {
  std::vector<hl_select_t> parsed_slices;

  // If user passed a single slice, convert it to a vector of slices
  std::vector<pybind_slice_item_t> items;
  for (int i = 0; i < slices.size(); i++) {
    // assert that object is a slice, int or list of ints
    py::object slice = slices[i];
    if (py::isinstance<py::slice>(slice)) {
      items.push_back(slice.cast<py::slice>());
    } else if (py::isinstance<py::int_>(slice)) {
      items.push_back(slice.cast<int>());
    } else if (py::isinstance<Tensor>(slice)) {
      items.push_back(slice.cast<Tensor>());
    } else if (py::isinstance<py::list>(slice) ||
               py::isinstance<py::array_t<int>>(slice)) {
      // To numpy array -> to tensor
      // check if the list contains only integers
      py::array_t<int> arr = slice.cast<py::array_t<int>>();
      Tensor t = Tensor::from_numpy(arr, inp_device);
      items.push_back(t);

    } else {
      throw std::runtime_error("[parse_pybind_slices] Invalid slice");
    }
  }

  PG_CHECK_ARG(items.size() <= arr_shape.size(),
               "Too many slices for the array, array ndim: ", arr_shape.size(),
               ", number of slices: ", items.size());
  for (int i = 0; i < items.size(); i++) {
    parsed_slices.push_back(parse_pybind_slice_item(items[i], arr_shape[i]));
  }

  return parsed_slices;
}

} // namespace pybind_utils
} // namespace pg