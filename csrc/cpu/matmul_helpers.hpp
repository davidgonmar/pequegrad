#include "dtype.hpp"
#include <cblas.h>
#include <vector>

// Assumes contiguous tensors
template <typename T>
void matmul_ker(const T *lhs, const T *rhs, T *result, size_t M, size_t N,
                size_t K, size_t B) {
#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < N; j++) {
        T sum = 0;
        for (size_t k = 0; k < K; k++) {
          sum += lhs[b * M * K + i * K + k] * rhs[b * K * N + k * N + j];
        }
        result[b * M * N + i * N + j] = sum;
      }
    }
  }
}

template <>
void matmul_ker<float>(const float *lhs, const float *rhs, float *result,
                       size_t M, size_t N, size_t K, size_t B) {
#pragma omp parallel for
  for (int b = 0; b < B; b++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                lhs + b * M * K, K, rhs + b * K * N, N, 0.0, result + b * M * N,
                N);
  }
}

template <>
void matmul_ker<double>(const double *lhs, const double *rhs, double *result,
                        size_t M, size_t N, size_t K, size_t B) {
#pragma omp parallel for
  for (int b = 0; b < B; b++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
                lhs + b * M * K, K, rhs + b * K * N, N, 0.0, result + b * M * N,
                N);
  }
}

template <typename T>
void dot_ker(const T *lhs, const T *rhs, T *result, size_t S) {
  T sum = 0;
  for (size_t i = 0; i < S; i++) {
    sum += lhs[i] * rhs[i];
  }
  *result = sum;
}

template <>
void dot_ker<float>(const float *lhs, const float *rhs, float *result,
                    size_t S) {
  *result = cblas_sdot(S, lhs, 1, rhs, 1);
}

template <>
void dot_ker<double>(const double *lhs, const double *rhs, double *result,
                     size_t S) {
  *result = cblas_ddot(S, lhs, 1, rhs, 1);
}

void dispatch_contiguous_matmul_ker(const void *lhs, const void *rhs,
                                    void *result, size_t M, size_t N, size_t K,
                                    size_t B, DType dtype) {
  switch (dtype) {
  case DType::Float32:
    matmul_ker<float>(static_cast<const float *>(lhs),
                      static_cast<const float *>(rhs),
                      static_cast<float *>(result), M, N, K, B);
    break;
  case DType::Float64:
    matmul_ker<double>(static_cast<const double *>(lhs),
                       static_cast<const double *>(rhs),
                       static_cast<double *>(result), M, N, K, B);
    break;
  case DType::Int32:
    matmul_ker<int32_t>(static_cast<const int32_t *>(lhs),
                        static_cast<const int32_t *>(rhs),
                        static_cast<int32_t *>(result), M, N, K, B);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

void dispatch_contiguous_dot_ker(const void *lhs, const void *rhs, void *result,
                                 size_t S, DType dtype) {
  switch (dtype) {
  case DType::Float32:
    dot_ker<float>(static_cast<const float *>(lhs),
                   static_cast<const float *>(rhs),
                   static_cast<float *>(result), S);
    break;
  case DType::Float64:
    dot_ker<double>(static_cast<const double *>(lhs),
                    static_cast<const double *>(rhs),
                    static_cast<double *>(result), S);
    break;
  case DType::Int32:
    dot_ker<int32_t>(static_cast<const int32_t *>(lhs),
                     static_cast<const int32_t *>(rhs),
                     static_cast<int32_t *>(result), S);
    break;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}
