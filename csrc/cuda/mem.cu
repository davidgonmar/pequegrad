#include "mem.hpp"

std::shared_ptr<void> allocate_cuda(const size_t nbytes) {
    void *ptr;
    cudaMalloc(&ptr, nbytes);
    return std::shared_ptr<void>(ptr, [](void *p) { cudaFree(p); });
}

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src, const std::shared_ptr<void> &dst, const size_t nbytes) {
    cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyHostToDevice);
}

void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src, const std::shared_ptr<void> &dst, const size_t nbytes) {
    cudaMemcpy(dst.get(), src.get(), nbytes, cudaMemcpyDeviceToHost);
}