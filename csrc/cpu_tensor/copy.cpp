

#include "dtype.hpp"
#include <vector>
#include "copy.hpp"
#include "shape.hpp"

namespace copy {

void dispatch_copy(const shape_t &shape, const shape_t &in_strides, const shape_t &out_strides, const void *in, void *out, DType dtype) {
  switch (dtype) {
  case DType::Float32:
    copy_ker<float>(shape, static_cast<const float *>(in),
                    static_cast<float *>(out), in_strides, out_strides);
    break;
  case DType::Float64:
    copy_ker<double>(shape, static_cast<const double *>(in),
                     static_cast<double *>(out), in_strides, out_strides);
    break;
  case DType::Int32:
    copy_ker<int32_t>(shape, static_cast<const int32_t *>(in),
                      static_cast<int32_t *>(out), in_strides, out_strides);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}
} // namespace copy