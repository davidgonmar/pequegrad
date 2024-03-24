#pragma once

#include <cmath>
#include <iostream>
#include "dtype.hpp"

/**
 * This functions implement unary element-wise operations on a vector.
 * They are vectorized using AVX if available, otherwise they fall back to regular loops.
*/
#ifdef __AVX__
#include <immintrin.h>
#define UNARY_OP_FLOAT(fn_name, avx_fn, fb_fn) \
static inline void fn_name(float *ptr, float *result, int size) { \
  int i; \
  for (i = 0; i < size - 7; i += 8) { \
    _mm256_storeu_ps(result + i, avx_fn(_mm256_loadu_ps(ptr + i))); \
  } \
  for (; i < size; i++) { \
    result[i] = fb_fn(ptr[i]); \
  } \
}

#define UNARY_OP_DOUBLE(fn_name, avx_fn, fb_fn) \
static inline void fn_name(double *ptr, double *result, int size) { \
  int i; \
  for (i = 0; i < size - 3; i += 4) { \
    _mm256_storeu_pd(result + i, avx_fn(_mm256_loadu_pd(ptr + i))); \
  } \
  for (; i < size; i++) { \
    result[i] = fb_fn(ptr[i]); \
  } \
}
#else
// Fallback implementation for non-AVX architectures
#define UNARY_OP_FLOAT(fn_name, avx_fn, fb_fn) \
static inline void fn_name(float *ptr, float *result, int size) { \
  for (int i = 0; i < size; i++) { \
    result[i] = fb_fn(ptr[i]); \
  } \
}

#define UNARY_OP_DOUBLE(fn_name, avx_fn, fb_fn) \
static inline void fn_name(double *ptr, double *result, int size) { \
  for (int i = 0; i < size; i++) { \
    result[i] = fb_fn(ptr[i]); \
  } \
}

#endif

UNARY_OP_FLOAT(vec_exp, _mm256_exp_ps, expf)
UNARY_OP_FLOAT(vec_log, _mm256_log_ps, logf)
UNARY_OP_DOUBLE(vec_exp, _mm256_exp_pd, exp)
UNARY_OP_DOUBLE(vec_log, _mm256_log_pd, log)

enum class UnaryOp {
  Exp,
  Log
};


static inline void dispatch_unary_op(DType dtype, UnaryOp op, void *ptr, void *result, int size) {
  switch (dtype) {
    case DType::Float32:
      switch (op) {
        case UnaryOp::Exp:
          vec_exp(static_cast<float *>(ptr), static_cast<float *>(result), size);
          break;
        case UnaryOp::Log:
          vec_log(static_cast<float *>(ptr), static_cast<float *>(result), size);
          break;
      }
      break;
    case DType::Int32:
        throw std::runtime_error("Unsupported data type: Int32");
        break;
    case DType::Float64:
        switch (op) {
            case UnaryOp::Exp:
            vec_exp(static_cast<double *>(ptr), static_cast<double *>(result), size);
            break;
            case UnaryOp::Log:
            vec_log(static_cast<double *>(ptr), static_cast<double *>(result), size);
            break;
        }
        break;
  }
}