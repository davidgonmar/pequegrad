#include "cuda_utils.cuh"
#include "mem.hpp"

std::shared_ptr<void> allocate_cuda(const size_t nbytes) {
  void *ptr;
  CHECK_CUDA(cudaMallocAsync(&ptr, nbytes, 0));
  return std::shared_ptr<void>(ptr, [](void *p) { CHECK_CUDA(cudaFree(p)); });
}

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyHostToDevice);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cpu_to_cuda(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src, nbytes, cudaMemcpyHostToDevice);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyDeviceToHost);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src, nbytes, cudaMemcpyDeviceToHost);
  CHECK_CUDA(cudaGetLastError());
}