#include "./binary.cuh"
#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dispatch.hpp"
#include "dtype.hpp"
#include "utils.hpp"
#include <cmath>
#include <numeric>

namespace pg {
namespace cuda {
DEF_BIN_OP_KERNEL(add_kernel, x + y, float)
DEF_BIN_OP_KERNEL(add_kernel, x + y, double)
DEF_BIN_OP_KERNEL(add_kernel, x + y, int)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, float)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, double)
DEF_BIN_OP_KERNEL(sub_kernel, x - y, int)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, float)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, double)
DEF_BIN_OP_KERNEL(mult_kernel, x *y, int)
DEF_BIN_OP_KERNEL(div_kernel, x / y, float)
DEF_BIN_OP_KERNEL(div_kernel, x / y, double)
DEF_BIN_OP_KERNEL(div_kernel, x / y, int)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, float)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, double)
DEF_BIN_OP_KERNEL(greater_kernel, x > y, int)
DEF_BIN_OP_KERNEL(less_kernel, x < y, float)
DEF_BIN_OP_KERNEL(less_kernel, x < y, double)
DEF_BIN_OP_KERNEL(less_kernel, x < y, int)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, float)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, double)
DEF_BIN_OP_KERNEL(equal_kernel, x == y, int)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, float)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, double)
DEF_BIN_OP_KERNEL(not_equal_kernel, x != y, int)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, float)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, double)
DEF_BIN_OP_KERNEL(greater_equal_kernel, x >= y, int)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, float)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, double)
DEF_BIN_OP_KERNEL(less_equal_kernel, x <= y, int)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, float)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, double)
DEF_BIN_OP_KERNEL(element_wise_max_kernel, x > y ? x : y, int)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), float)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), double)
DEF_BIN_OP_KERNEL(pow_kernel, pow(x, y), int)

} // namespace cuda

#define DEF_BINARY_OP(NAME, kernel_name)                                       \
  void NAME::dispatch_cuda(const std::vector<Tensor> &inputs,                  \
                           std::vector<Tensor> &outputs) {                     \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].view_ptr()->allocate();                                         \
    size_t numels = a.numel();                                                 \
    auto d_strides_a =                                                         \
        cuda_unique_ptr_from_host<stride_t>(a.ndim(), a.strides().data());     \
    auto d_strides_b =                                                         \
        cuda_unique_ptr_from_host(b.ndim(), b.strides().data());               \
    auto d_shape = cuda_unique_ptr_from_host(a.ndim(), a.shape().data());      \
    size_t num_dims = a.ndim();                                                \
    size_t total_dynamic_smem =                                                \
        sizeof(stride_t) * num_dims * 2 + sizeof(size_t) * num_dims;           \
    PG_DISPATCH_ALL_TYPES(a.dtype(), #NAME, [&]() {                            \
      dim3 blocksize(DEFAULT_BLOCK_SIZE);                                      \
      dim3 gridsize((numels + blocksize.x - 1) / blocksize.x);                 \
      cuda::kernel_name<<<gridsize, blocksize, total_dynamic_smem>>>(          \
          d_strides_a.get(), d_strides_b.get(), d_shape.get(), a.ndim(),       \
          a.get_casted_base_ptr<scalar_t>(),                                   \
          b.get_casted_base_ptr<scalar_t>(),                                   \
          outputs[0].get_casted_base_ptr<scalar_t>());                         \
    });                                                                        \
    PG_CUDA_KERNEL_END;                                                        \
  }
DEF_BINARY_OP(Add, add_kernel)
DEF_BINARY_OP(Pow, pow_kernel)
DEF_BINARY_OP(Sub, sub_kernel)
DEF_BINARY_OP(Max, element_wise_max_kernel)
DEF_BINARY_OP(Mul, mult_kernel)
DEF_BINARY_OP(Div, div_kernel)
DEF_BINARY_OP(Gt, greater_kernel)
DEF_BINARY_OP(Lt, less_kernel)
DEF_BINARY_OP(Eq, equal_kernel)
DEF_BINARY_OP(Neq, not_equal_kernel)
DEF_BINARY_OP(Ge, greater_equal_kernel)
DEF_BINARY_OP(Le, less_equal_kernel)
} // namespace pg