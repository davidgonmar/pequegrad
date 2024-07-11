#include "./binary.cuh"
#include "./unary.cuh"
#include "ad_primitives.hpp"
#include "common/view_helpers.hpp"
#include "cuda_utils.cuh"
#include "random.cuh"
#include "tensor.hpp"
#include "view_helpers.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

namespace pg {

void Reshape::dispatch_cuda(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  shape_t shape = inputs[0].shape();
  Tensor a = inputs[0];
  axes_t _new_shape = _shape_to;
  shape_t new_shape(_new_shape.size());
  size_t total_new = 1;

  int neg_pos = -1;
  for (size_t i = 0; i < _new_shape.size(); i++) {
    if (_new_shape[i] < 0) {
      PG_CHECK_ARG(
          neg_pos == -1,
          "Can only specify one unknown dimension (-1) for reshape, got ",
          neg_pos, " and ", i, " for shape ", vec_to_string(_new_shape));
      neg_pos = i;
    }
    new_shape[i] = _new_shape[i];
    total_new *= new_shape[i] == -1 ? 1 : new_shape[i];
  }

  size_t total_old =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  if (neg_pos != -1) {
    new_shape[neg_pos] = total_old / total_new;
    PG_CHECK_ARG(
        total_old % total_new == 0,
        "New shape is not compatible with old shape: ", vec_to_string(shape),
        " not compatible with ", vec_to_string(_new_shape));
  }
  total_new = total_old;

  if (a.is_contiguous()) {
    outputs[0].init_view(std::make_shared<View>(
        view::nocopy_reshape_nocheck(a.view(), new_shape)));
    return;
  } else {
    View cont_view = cuda::view::as_contiguous(a.view());
    outputs[0].init_view(std::make_shared<View>(
        view::nocopy_reshape_nocheck(cont_view, new_shape)));
    return;
  }
}

void AsContiguous::dispatch_cuda(const std::vector<Tensor> &inputs,
                                 std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  outputs[0].init_view(
      std::make_shared<View>(cuda::view::as_contiguous(inputs[0].view())));
}

void AsType::dispatch_cuda(const std::vector<Tensor> &inputs,
                           std::vector<Tensor> &outputs) {
  CHECK_INPUTS_LENGTH(inputs, 1);
  CHECK_OUTPUTS_LENGTH(outputs, 1);
  const Tensor &a = inputs[0];
  outputs[0].init_view(
      std::make_shared<View>(cuda::view::astype(a.view(), _dtype_to)));
}

} // namespace pg