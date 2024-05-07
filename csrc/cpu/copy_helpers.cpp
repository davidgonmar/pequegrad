

#include "copy_helpers.hpp"
#include "dtype.hpp"
#include "shape.hpp"
#include <vector>

namespace copy {

void dispatch_copy(const shape_t &shape, const strides_t &in_strides,
                   const strides_t &out_strides, const void *in, void *out,
                   DType dtype) {
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

void dispatch_cast(const shape_t &shape, const strides_t &in_strides,
                   const strides_t &out_strides, const void *in, void *out,
                   DType in_dtype, DType out_dtype) {
  switch (in_dtype) {
  case DType::Float32:
    switch (out_dtype) {
    case DType::Float32:
      cast_ker<float, float>(shape, static_cast<const float *>(in),
                             static_cast<float *>(out), in_strides,
                             out_strides);
      break;
    case DType::Float64:
      cast_ker<float, double>(shape, static_cast<const float *>(in),
                              static_cast<double *>(out), in_strides,
                              out_strides);
      break;
    case DType::Int32:
      cast_ker<float, int32_t>(shape, static_cast<const float *>(in),
                               static_cast<int32_t *>(out), in_strides,
                               out_strides);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
    break;
  case DType::Float64:
    switch (out_dtype) {
    case DType::Float32:
      cast_ker<double, float>(shape, static_cast<const double *>(in),
                              static_cast<float *>(out), in_strides,
                              out_strides);
      break;
    case DType::Float64:
      cast_ker<double, double>(shape, static_cast<const double *>(in),
                               static_cast<double *>(out), in_strides,
                               out_strides);
      break;
    case DType::Int32:
      cast_ker<double, int32_t>(shape, static_cast<const double *>(in),
                                static_cast<int32_t *>(out), in_strides,
                                out_strides);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
    break;
  case DType::Int32:
    switch (out_dtype) {
    case DType::Float32:
      cast_ker<int32_t, float>(shape, static_cast<const int32_t *>(in),
                               static_cast<float *>(out), in_strides,
                               out_strides);
      break;
    case DType::Float64:
      cast_ker<int32_t, double>(shape, static_cast<const int32_t *>(in),
                                static_cast<double *>(out), in_strides,
                                out_strides);
      break;
    case DType::Int32:
      cast_ker<int32_t, int32_t>(shape, static_cast<const int32_t *>(in),
                                 static_cast<int32_t *>(out), in_strides,
                                 out_strides);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}
} // namespace copy
