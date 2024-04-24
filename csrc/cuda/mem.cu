#include "cuda_tensor/cuda_utils.cuh"
#include "mem.hpp"

static size_t alloc_count = 0;
static size_t free_count = 0;
std::shared_ptr<void> allocate_cuda(const size_t nbytes) {
  void *ptr;
  CHECK_CUDA(cudaMalloc(&ptr, nbytes));
  alloc_count++;
  return std::shared_ptr<void>(ptr, [](void *p) {
    CHECK_CUDA(cudaFree(p));
    free_count++;
  });
}

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyHostToDevice);
}

void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyDeviceToHost);
}