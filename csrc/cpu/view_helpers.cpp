#include "view_helpers.hpp"
#include "dispatch.hpp"

namespace pg {
namespace cpu {
namespace view {
View as_contiguous(const View &view, bool force) {
  if (view.is_contiguous() && !force) {
    return view;
  }
  View new_view = View(view.shape(), view.dtype(), device::from_str("cpu"));
  PG_DISPATCH_ALL_TYPES(view.dtype(), "dispatch_copy", [&] {
    copy_ker<scalar_t>(view.shape(), view.get_casted_base_ptr<scalar_t>(),
                       new_view.get_casted_base_ptr<scalar_t>(), view.strides(),
                       new_view.strides());
  });
  return new_view;
}
View astype(const View &view, DType dtype) {
  if (view.dtype() == dtype) {
    return view;
  }
  View new_view = View(view.shape(), dtype, device::from_str("cpu"));
  PG_DISPATCH_ALL_TYPES_TWO_TYPES(
      view.dtype(), new_view.dtype(), "dispatch_cast", [&] {
        cast_ker<scalar_t1, scalar_t2>(
            view.shape(), view.get_casted_base_ptr<scalar_t1>(),
            new_view.get_casted_base_ptr<scalar_t2>(), view.strides(),
            new_view.strides());
      });
  return new_view;
}
void copy_data(const View &view, View &dst) {
  PG_DISPATCH_ALL_TYPES(view.dtype(), "dispatch_copy", [&] {
    copy_ker<scalar_t>(view.shape(), view.get_casted_base_ptr<scalar_t>(),
                       dst.get_casted_base_ptr<scalar_t>(), view.strides(),
                       dst.strides());
  });
}
} // namespace view
} // namespace cpu
} // namespace pg