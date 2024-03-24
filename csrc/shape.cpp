#include "shape.hpp"
#include "utils.hpp"

shape_t get_strides_for_broadcasting(const shape_t shape_from, const shape_t strides_from,
                                          const shape_t shape_to) {

  const size_t from_ndim = shape_from.size();
  const size_t to_ndim = shape_to.size();
  // Cannot broadcast if the number of dimensions of the from array is greater
  // than the number of dimensions of the to array
  PG_CHECK_ARG(from_ndim <= to_ndim,
               "from_ndim must be <= to_ndim, trying to broadcast from ",
               vec_to_string(shape_from), " to ", vec_to_string(shape_to));

  int new_size = 1;
  shape_t new_strides(to_ndim, 0);
  // reverse test if the dim is 1 or they are equal
  for (int i = to_ndim - 1, j = from_ndim - 1; i >= 0; --i, --j) {
    size_t dim_to = shape_to[i];
    size_t dim_from =
        (j >= 0) ? shape_from[j]
                 : -1; // -1 means we 'ran' out of dimensions for j

    PG_CHECK_ARG(dim_to == dim_from || dim_from == 1 || dim_from == -1,
                 "got incompatible shapes: ", vec_to_string(shape_from),
                 " cannot be broadcasted to ", vec_to_string(shape_to),
                 ". In dimension ", i, " got dim_to=", dim_to,
                 " and dim_from=", dim_from);

    if (dim_from != 1 && dim_from != -1) {
      new_strides[i] = strides_from[j];
    }
    new_size *= dim_to;
  }
  
return new_strides;
}
