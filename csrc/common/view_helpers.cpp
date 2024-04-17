#include "view_helpers.hpp"

// returns new strides + axes of the output that are 'broadcasted'
static std::tuple<strides_t, axes_t>
get_broadcasting_info(const shape_t shape_from, const strides_t strides_from,
                      const shape_t shape_to) {

  const size_t from_ndim = shape_from.size();
  const size_t to_ndim = shape_to.size();
  // Cannot broadcast if the number of dimensions of the from array is greater
  // than the number of dimensions of the to array
  PG_CHECK_ARG(from_ndim <= to_ndim,
               "from_ndim must be <= to_ndim, trying to broadcast from ",
               vec_to_string(shape_from), " to ", vec_to_string(shape_to));

  int new_size = 1;
  strides_t new_strides(to_ndim, 0);
  axes_t broadcasted_axes;
  // reverse test if the dim is 1 or they are equal
  for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
    size_t dim_to = shape_to[i];
    size_t dim_from = (j >= 0)
                          ? shape_from[j]
                          : -1; // -1 means we 'ran' out of dimensions for j

    PG_CHECK_ARG(dim_to == dim_from || dim_from == 1 || dim_from == -1,
                 "got incompatible shapes: ", vec_to_string(shape_from),
                 " cannot be broadcasted to ", vec_to_string(shape_to),
                 ". In dimension ", i, " got dim_to=", dim_to,
                 " and dim_from=", dim_from);

    if (dim_from != 1 && dim_from != -1) {
      new_strides[i] = strides_from[j];
    } else {
      broadcasted_axes.push_back(i);
    }
    new_size *= dim_to;
  }

  return std::make_tuple(new_strides, broadcasted_axes);
}

namespace pg {
namespace view {
std::tuple<View, axes_t> broadcasted_to(const View &view,
                                        const shape_t &shape_to) {
  if (view.shape() == shape_to) {
    return std::make_tuple(view, axes_t());
  }
  auto [new_strides, broadcasted_axes] =
      get_broadcasting_info(view.shape(), view.strides(), shape_to);
  return std::make_tuple(View(view.shared_ptr(), view.nbytes(), shape_to,
                              new_strides, view.offset(), view.dtype(),
                              view.device()),
                         broadcasted_axes);
}
View squeeze(const View &orig, axis_t axis) {
  if (axis < 0) {
    axis = orig.shape().size() + axis;
  }
  PG_CHECK_ARG(axis < orig.shape().size(),
               "[view::squeeze] axis out of bounds, got ", axis, " for shape ",
               vec_to_string(orig.shape()));
  PG_CHECK_ARG(orig.shape()[axis] == 1, "[view:squeeze] cannot squeeze axis ",
               axis, " as it is not 1, got ", orig.shape()[axis]);

  shape_t new_shape;
  strides_t new_strides;
  for (size_t i = 0; i < orig.shape().size(); i++) {
    if (i != axis) {
      new_shape.push_back(orig.shape()[i]);
      new_strides.push_back(orig.strides()[i]);
    }
  }
  return View(orig.shared_ptr(), orig.nbytes(), new_shape, new_strides,
              orig.offset(), orig.dtype(), orig.device());
}
View squeeze(const View &orig, const axes_t &axes) {
  View view = orig;
  // we need to sort them in reverse order
  axes_t processed_nonnegative_axes;
  for (auto axis : axes) {
    if (axis < 0) {
      axis = orig.shape().size() + axis;
    }
    processed_nonnegative_axes.push_back(axis);
  }
  std::sort(processed_nonnegative_axes.begin(),
            processed_nonnegative_axes.end());
  std::reverse(processed_nonnegative_axes.begin(),
               processed_nonnegative_axes.end());
  shape_t new_shape;
  strides_t new_strides;
  for (auto axis : processed_nonnegative_axes) {
    view = squeeze(view, axis);
  }
  return view;
}
View squeeze(const View &orig) {
  axes_t all_axes;
  for (long i = orig.shape().size() - 1; i >= 0; i--) {
    if (orig.shape()[i] == 1) {
      all_axes.push_back(i);
    }
  }
  return squeeze(orig, all_axes);
}

View unsqueeze(const View &orig, axis_t axis) {
  if (axis < 0) {
    axis = orig.shape().size() + axis + 1;
  }
  PG_CHECK_ARG(axis <= orig.shape().size(),
               "[view::unsqueeze] axis out of bounds, got ", axis,
               " for shape ", vec_to_string(orig.shape()), " of size ",
               orig.shape().size());
  shape_t new_shape(orig.shape());
  new_shape.insert(new_shape.begin() + axis, 1);
  strides_t new_strides(orig.strides());
  new_strides.insert(new_strides.begin() + axis,
                     (axis < orig.strides().size())
                         ? orig.strides()[std::max(0, (int)axis - 1)]
                         : dtype_to_size(orig.dtype()));
  return View(orig.shared_ptr(), orig.nbytes(), new_shape, new_strides,
              orig.offset(), orig.dtype(), orig.device());
}
View unsqueeze(const View &orig, const axes_t &axes) {
  View view = orig;
  // we need to sort them in reverse order
  axes_t processed_nonnegative_axes;
  for (auto axis : axes) {
    if (axis < 0) {
      axis = orig.shape().size() + axis;
    }
    processed_nonnegative_axes.push_back(axis);
  }
  std::sort(processed_nonnegative_axes.begin(),
            processed_nonnegative_axes.end());
  std::cout << "axes: " << vec_to_string(processed_nonnegative_axes)
            << std::endl;
  // we dont reverse in unsqueeze
  for (auto axis : processed_nonnegative_axes) {
    view = unsqueeze(view, axis);
  }
  return view;
}

View permute(const View &orig, const axes_t &axes) {
  // TODO -- maybe check that the axes are indeed correct
  PG_CHECK_ARG(axes.size() == orig.shape().size(),
               "[view::permute] axes size must be equal to shape size, got ",
               axes.size(), " and ", orig.shape().size());
  shape_t new_shape;
  strides_t new_strides;
  for (size_t i = 0; i < axes.size(); i++) {
    axis_t axis = axes[i] >= 0 ? axes[i] : orig.shape().size() + axes[i];
    PG_CHECK_ARG(axis < orig.shape().size(),
                 "[view::permute] axis out of bounds, got ", axis,
                 " for shape ", vec_to_string(orig.shape()));
    new_shape.push_back(orig.shape()[axis]);
    new_strides.push_back(orig.strides()[axis]);
  }
  return View(orig.shared_ptr(), orig.nbytes(), new_shape, new_strides,
              orig.offset(), orig.dtype(), orig.device());
}
} // namespace view
} // namespace pg