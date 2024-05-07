#include "view_helpers.hpp"
#include "./copy_helpers.hpp"

namespace pg {
namespace cpu {
namespace view {
View as_contiguous(const View &view, bool force) {
  if (view.is_contiguous() && !force) {
    return view;
  }
  View new_view = View(view.shape(), view.dtype(), device::CPU);
  copy::dispatch_copy(view.shape(), view.strides(), new_view.strides(),
                      view.get_base_ptr(), new_view.get_base_ptr(),
                      view.dtype());
  return new_view;
}
View astype(const View &view, DType dtype) {
  if (view.dtype() == dtype) {
    return view;
  }
  View new_view = View(view.shape(), dtype, device::CPU);
  copy::dispatch_cast(view.shape(), view.strides(), new_view.strides(),
                      view.get_base_ptr(), new_view.get_base_ptr(),
                      view.dtype(), dtype);
  return new_view;
}
} // namespace view
} // namespace cpu
} // namespace pg